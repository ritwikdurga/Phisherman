import re
import socket
import ssl
import whois
import requests
from urllib.parse import urlparse
from googlesearch import search
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import tldextract
import dns.resolver
import time

requests.packages.urllib3.disable_warnings()

TRUSTED_CAS = [
    "GeoTrust", "GoDaddy", "Network Solutions", "Thawte", "Comodo", "Doster", "VeriSign",
    "Let's Encrypt", "DigiCert", "GlobalSign", "Entrust", "Symantec", "RapidSSL", "SSL.com",
    "Google Trust Services"
]

# Helper Functions (unchanged)
def is_ip_address(hostname):
    try:
        socket.inet_aton(hostname)
        return True
    except socket.error:
        return False

def get_page_content_and_history(url):
    try:
        response = requests.get(url, timeout=10, verify=False, allow_redirects=True)
        return response.text, response.history
    except requests.exceptions.ConnectTimeout:
        print(f"Warning: Connection to {url} timed out after 10 seconds.")
        return None, []
    except requests.exceptions.RequestException as e:
        print(f"Warning: Unable to access {url} - {e}")
        return None, []

# Address Bar Based Features
def having_ip_address(url):
    match = re.search(r'\d+\.\d+\.\d+\.\d+', url)
    return (-1, "IP address in URL (phishing)") if match else (1, "No IP address")

def url_length(url):
    length = len(url)
    if length >= 54:
        return -1, f"Long URL ({length} chars, phishing)"
    return 1, f"Normal URL length ({length} chars)"

def shortening_service(url):
    shorteners = r'(bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|is\.gd|t\.co|adf\.ly)'
    return (-1, "Shortening service detected") if re.search(shorteners, url) else (1, "No shortening service")

def having_at_symbol(url):
    return (-1, "@ symbol in URL (phishing)") if '@' in url else (1, "No @ symbol")

def double_slash_redirecting(url):
    pos = url.find('//', 8)
    return (-1, "Suspicious // redirection") if pos != -1 else (1, "No suspicious redirection")

def prefix_suffix(url):
    domain = tldextract.extract(url).domain
    return (-1, "Prefix/Suffix (-) in domain") if '-' in domain else (1, "No prefix/suffix")

def sub_domains(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    if not subdomain:
        return 1, "No subdomains"
    sub_parts = subdomain.split('.')
    if sub_parts[0].lower() == 'www':
        sub_parts = sub_parts[1:]
    dots = len(sub_parts) - 1 if sub_parts else 0
    if dots == 0:
        return 1, "No additional subdomains"
    elif dots == 1:
        return 0, "One subdomain (suspicious)"
    else:
        return -1, "Multiple subdomains (phishing)"

def https_certificate(url):
    if not url.startswith('https'):
        return -1, "No HTTPS protocol"
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
        issuer = dict(x[0] for x in cert['issuer']).get('organizationName', '')
        trusted = any(ca in issuer for ca in TRUSTED_CAS)
        cert_date_str = cert.get('notAfter')
        if not cert_date_str:
            return -1, "Certificate has no expiration date"
        cert_date = datetime.strptime(cert_date_str, '%b %d %H:%M:%S %Y %Z').replace(tzinfo=timezone.utc)
        days_valid = (cert_date - datetime.now(timezone.utc)).days
        # Adjust to flag short-term as phishing (-1) per document
        if trusted and days_valid >= 730:
            return 1, f"Trusted CA ({issuer}), expires in {days_valid} days"
        elif trusted:
            return -1, f"Trusted CA ({issuer}), short-term ({days_valid} days)"
        else:
            return -1, f"Untrusted CA ({issuer})"
    except Exception as e:
        return -1, f"Certificate check failed: {e}"

def domain_registration_length(domain):
    if is_ip_address(domain):
        return -1, "IP address, no registration data"
    try:
        w = whois.whois(domain)
        if w.expiration_date:
            exp_date = min(w.expiration_date) if isinstance(w.expiration_date, list) else w.expiration_date
            if exp_date.tzinfo is None:
                exp_date = exp_date.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_left = (exp_date - now).days
            return (1, f"Registered for {days_left} days") if days_left > 365 else (-1, "Short registration")
        return -1, "No expiration date"
    except Exception as e:
        return -1, f"WHOIS failed: {e}"

def favicon_check(url, soup):
    if soup is None:
        return 0, "Page inaccessible, cannot verify favicon"
    try:
        favicon = soup.find('link', rel=lambda x: x in ['shortcut icon', 'icon'] if x else False)
        if favicon and favicon.get('href'):
            favicon_netloc = urlparse(favicon['href']).netloc or urlparse(url).netloc
            if favicon_netloc != urlparse(url).netloc:
                return -1, "External favicon"
        return 1, "Favicon local or absent"
    except Exception as e:
        return -1, f"Favicon check failed: {e}"

def non_standard_port(url):
    parsed = urlparse(url)
    port = parsed.port
    if port and port not in [80, 443]:
        return -1, f"Non-standard port: {port}"
    return 1, "Standard port"

def https_domain_token(url):
    domain = tldextract.extract(url).domain
    if 'https' in domain.lower():
        return -1, "'HTTPS' in domain (phishing)"
    return 1, "No 'HTTPS' in domain"

# Abnormal Based Features
def request_url(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    tags = soup.find_all(['img', 'video', 'audio', 'script', 'link'])
    if not tags:
        return 1, "No external resources"
    external = sum(1 for t in tags for attr in ['src', 'href'] if t.get(attr) and urlparse(t.get(attr)).netloc and urlparse(t.get(attr)).netloc != urlparse(url).netloc)
    ratio = (external / len(tags)) * 100
    return (-1, f"High external resources: {ratio:.1f}%") if ratio > 61 else (1, f"External resources: {ratio:.1f}%")

def url_of_anchor(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    anchors = soup.find_all('a')
    if not anchors:
        return 1, "No anchors"
    invalid = sum(1 for a in anchors if a.get('href', '').startswith(('#', 'javascript:', '')))
    external = sum(1 for a in anchors if a.get('href') and urlparse(a.get('href')).netloc and urlparse(a.get('href')).netloc != urlparse(url).netloc)
    total_suspicious = invalid + external
    ratio = min((total_suspicious / len(anchors)) * 100, 100)
    return (-1, f"High suspicious anchors: {ratio:.1f}%") if ratio > 67 else (1, f"Suspicious anchors: {ratio:.1f}%")

def links_in_tags(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    tags = soup.find_all(['meta', 'script', 'link'])
    if not tags:
        return 1, "No such tags"
    external = sum(1 for t in tags for attr in ['src', 'href'] if t.get(attr) and urlparse(t.get(attr)).netloc and urlparse(t.get(attr)).netloc != urlparse(url).netloc)
    ratio = (external / len(tags)) * 100
    return (-1, f"High external links: {ratio:.1f}%") if ratio > 61 else (1, f"External links: {ratio:.1f}%")

def server_form_handler(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    forms = soup.find_all('form')
    if not forms:
        return 1, "No forms"
    for form in forms:
        action = form.get('action', '').strip().lower()
        if not action or action in ['about:blank', '#']:
            return -1, "Empty or blank form action"
        if urlparse(action).netloc and urlparse(action).netloc != urlparse(url).netloc:
            return 0, "External form handler"
    return 1, "Local form handlers"

def submitting_to_email(content):
    if content is None:
        return -1, "Page inaccessible"
    if re.search(r'mail\(|mailto:', content, re.IGNORECASE):
        return -1, "Email submission detected"
    return 1, "No email submission"

def abnormal_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    extracted = tldextract.extract(url)
    registered_domain = extracted.registered_domain
    if is_ip_address(hostname) or hostname in [registered_domain, f"www.{registered_domain}"]:
        return 1, "URL matches domain"
    return -1, "Abnormal URL structure"

# HTML and JavaScript Based Features
def website_forwarding(history):
    num_redirects = len(history)
    if num_redirects <= 1:
        return 1, f"{num_redirects} redirects (legitimate)"
    elif num_redirects >= 4:
        return -1, f"{num_redirects} redirects (phishing)"
    else:
        return 0, f"{num_redirects} redirects (suspicious)"

def status_bar_customization(content):
    if content is None:
        return -1, "Page inaccessible"
    if 'onmouseover' in content.lower():
        return -1, "Status bar modification"
    return 1, "No status bar changes"

def disabling_right_click(content):
    if content is None:
        return -1, "Page inaccessible"
    if 'event.button==2' in content.lower():
        return -1, "Right-click disabled"
    return 1, "Right-click enabled"

def using_popup_window(content):
    if content is None:
        return -1, "Page inaccessible"
    if 'window.open(' in content.lower():
        return -1, "Popup detected"
    return 1, "No popups"

def iframe_redirection(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    iframes = soup.find_all('iframe')
    if not iframes:
        return 1, "No iframes"
    suspicious = sum(1 for i in iframes if i.get('frameborder', '').lower() == '0' or (i.get('src') and urlparse(i.get('src')).netloc != urlparse(url).netloc))
    return -1, f"{len(iframes)} iframes, {suspicious} suspicious" if suspicious else (0, f"{len(iframes)} local iframes")

# Domain Based Features
def age_of_domain(domain):
    if is_ip_address(domain):
        return -1, "IP address, no domain age"
    try:
        w = whois.whois(domain)
        creation_date = min(w.creation_date) if isinstance(w.creation_date, list) else w.creation_date
        if creation_date:
            if creation_date.tzinfo is None:
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            age_days = (now - creation_date).days
            return (1, f"Age: {age_days} days") if age_days >= 180 else (-1, "Age < 6 months")
        return -1, "No creation date"
    except Exception as e:
        return -1, f"WHOIS failed: {e}"

def dns_record(domain):
    if is_ip_address(domain):
        return 1, "IP address, DNS assumed"
    try:
        answers = dns.resolver.resolve(domain, 'A')
        return 1, f"{len(answers)} DNS records"
    except Exception:
        return -1, "No DNS records"

def website_traffic(url):
    """
    Check website traffic by leveraging the google_index result.
    Returns 1 if indexed (assumed traffic), -1 if not indexed (low traffic).
    """
    domain = urlparse(url).netloc
    query = f"site:{domain}"

    try:
        results = list(search(query)) 
        indexed_pages = len(results)

        if indexed_pages > 100:
            return 1, f"High traffic ({indexed_pages} pages indexed)"
        elif indexed_pages > 10:
            return 0, f"Moderate traffic ({indexed_pages} pages indexed)"
        else:
            return -1, "Low traffic (very few pages indexed)"

    except Exception as e:
        return -1, f"Google search failed: {str(e)}"

def page_rank(url):
    score, message = google_index(url)
    if score == 1:
        return 1, "Likely reputable (indexed by Google)"
    return -1, "Low authority (not indexed or error)"

def google_index(url):
    try:
        query = f"site:{urlparse(url).netloc}"
        results = search(query)
        indexed = any(results)
        return (1, "Indexed by Google") if indexed else (-1, "Not indexed")
    except Exception as e:
        return -1, f"Google index check failed: {e}"

def number_of_links_pointing(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    links = soup.find_all('a', href=True)
    external = sum(1 for a in links if urlparse(a['href']).netloc and urlparse(a['href']).netloc != urlparse(url).netloc)
    if external >= 2:
        return 1, f"{external} outbound external links (not inbound - limitation)"
    return -1, "Few or no outbound external links"

# VirusTotal Check Function (unchanged but with added comment)
def check_virustotal(url):
    # Replace "API_KEY_HERE" with a valid VirusTotal API key from https://www.virustotal.com/gui/home/search
    API_KEY = "API_KEY_HERE"
    headers = {
        "x-apikey": API_KEY,
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded"
    }
    data = {"url": url}
    try:
        response = requests.post("https://www.virustotal.com/api/v3/urls", headers=headers, data=data, timeout=10)
        response.raise_for_status()
        analysis_id = response.json()["data"]["id"]
        for _ in range(10):
            response = requests.get(f"https://www.virustotal.com/api/v3/analyses/{analysis_id}", headers=headers, timeout=10)
            response.raise_for_status()
            analysis = response.json()["data"]["attributes"]
            if analysis["status"] == "completed":
                stats = analysis["stats"]
                if stats["malicious"] > 0:
                    return -1, f"Flagged by VirusTotal: {stats['malicious']} engines detect malicious"
                return 1, "Not flagged by VirusTotal"
            time.sleep(5)
        return 0, "VirusTotal analysis timed out after 50 seconds"
    except requests.exceptions.RequestException as e:
        return 0, f"VirusTotal check failed: {str(e)}"

# Main Analysis Function (unchanged)
def run_checks(url):
    extracted = tldextract.extract(url)
    domain = extracted.registered_domain or extracted.domain
    content, history = get_page_content_and_history(url)
    soup = BeautifulSoup(content, 'html.parser') if content else None

    features = {
        "Using IP Address": having_ip_address(url),
        "Long URL": url_length(url),
        "Shortening Service": shortening_service(url),
        "Having @ Symbol": having_at_symbol(url),
        "Redirecting //": double_slash_redirecting(url),
        "Prefix/Suffix (-)": prefix_suffix(url),
        "Sub Domains": sub_domains(url),
        "HTTPS Certificate": https_certificate(url),
        "Domain Registration": domain_registration_length(domain),
        "Favicon": favicon_check(url, soup),
        "Non-Standard Port": non_standard_port(url),
        "HTTPS in Domain": https_domain_token(url),
        "Request URL": request_url(url, soup),
        "URL of Anchor": url_of_anchor(url, soup),
        "Links in Tags": links_in_tags(url, soup),
        "Server Form Handler": server_form_handler(url, soup),
        "Email Submission": submitting_to_email(content),
        "Abnormal URL": abnormal_url(url),
        "Website Forwarding": website_forwarding(history),
        "Status Bar Mod": status_bar_customization(content),
        "Right Click Disable": disabling_right_click(content),
        "Popup Windows": using_popup_window(content),
        "Iframe Redirection": iframe_redirection(url, soup),
        "Age of Domain": age_of_domain(domain),
        "DNS Record": dns_record(domain),
        "Website Traffic": website_traffic(url),
        "PageRank": page_rank(url),
        "Google Index": google_index(url),
        "Links Pointing": number_of_links_pointing(url, soup),
        "Phishing Reports": check_virustotal(url)
    }

    print(f"\nSecurity Analysis for: {url}\n{'=' * 50}")
    for feature, (score, reason) in features.items():
        status = "SAFE" if score == 1 else "WARNING" if score == 0 else "DANGER"
        print(f"{feature:25} [{status:^7}] {reason}")

if __name__ == "__main__":
    test_url = "https://www.facebook.com"
    run_checks(test_url)