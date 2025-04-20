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
# 1. Using IP Address
def having_ip_address(url):
    match = re.search(r'\d+\.\d+\.\d+\.\d+', url)
    return (-1, "IP address in URL (phishing)") if match else (1, "No IP address")

# 2. Long URL to Hide the Suspicious Part
def url_length(url):
    length = len(url)
    if length >= 54:
        return -1, f"Long URL ({length} chars, phishing)"
    return 1, f"Normal URL length ({length} chars)"

# 3. URL Shortening Service
def shortening_service(url):
    shorteners = r'(bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|is\.gd|t\.co|adf\.ly)'
    return (-1, "Shortening service detected") if re.search(shorteners, url) else (1, "No shortening service")

# 4. Having @ Symbol which leads the browser to ignore everything preceding the “@” symbol
#    and the real address often follows the “@” symbol.
def having_at_symbol(url):
    return (-1, "@ symbol in URL (phishing)") if '@' in url else (1, "No @ symbol")

# 5. Redirection "//" in URL
def double_slash_redirecting(url):
    pos = url.find('//', 8)
    return (-1, "Suspicious // redirection") if pos != -1 else (1, "No suspicious redirection")

# 6. Adding Prefix or Suffix separated by '-' to the Domain
def prefix_suffix(url):
    domain = tldextract.extract(url).domain
    return (-1, "Prefix/Suffix (-) in domain") if '-' in domain else (1, "No prefix/suffix")

# 7. Subdomains and Multi Subdomains
def sub_domains(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    if not subdomain or subdomain.lower() == 'www':
        return 1, "No subdomains (Legitimate)"
    sub_parts = subdomain.split('.')
    if sub_parts[0].lower() == 'www':
        sub_parts = sub_parts[1:]
    dots = len(sub_parts)
    if dots <= 1:
        return 1, "No additional subdomains (Legitimate)"
    elif dots == 2:
        return 0, "One subdomain (Suspicious)"
    else:
        return -1, "Multiple subdomains (Phishing)"

# 8. HTTPS Certificate
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
        trusted = any(ca.lower() in issuer.lower() for ca in TRUSTED_CAS)
        cert_date_str = cert.get('notAfter')
        if not cert_date_str:
            return -1, "Certificate has no expiration date"
        cert_date = datetime.strptime(cert_date_str, '%b %d %H:%M:%S %Y %Z').replace(tzinfo=timezone.utc)
        days_valid = (cert_date - datetime.now(timezone.utc)).days
        if trusted and days_valid >= 365:
            return 1, f"Trusted CA ({issuer}), expires in {days_valid} days"
        elif trusted:
            return 0, f"Trusted CA ({issuer}), short-term ({days_valid} days)"
        else:
            return -1, f"Untrusted CA ({issuer})"
    except Exception as e:
        return -1, f"Certificate check failed: {e}"

# 9. Domain Registration Length (WHOIS) - Check if the domain is registered for a long time
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
            print(f"Domain: {domain}, Expiration Date: {exp_date}, Days Left: {days_left}")
            return (1, f"Registered for {days_left} days") if days_left > 365 else (-1, "Short registration")
        return -1, "No expiration date"
    except Exception as e:
        return -1, f"WHOIS failed: {e}"

# 10. Favicon - Check if the favicon is local or external
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

# 11. Non-standard Port - Check if the port is not standard (80, 443)
def non_standard_port(url):
    parsed = urlparse(url)
    port = parsed.port
    if port and port not in [80, 443]:
        return -1, f"Non-standard port: {port}"
    return 1, "Standard port"

# 12. HTTPS in Domain Part - Check if the domain part has 'HTTPS' in it
#     The phishers may add the “HTTPS” token to the domain part of a URL in order to trick users.
def https_domain_token(url):
    domain = tldextract.extract(url).domain
    if 'https' in domain.lower():
        return -1, "'HTTPS' in domain (phishing)"
    return 1, "No 'HTTPS' in domain"

# Abnormal Based Features
# 13. External Resources - Check the ratio of external resources to total resources
def request_url(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    tags = soup.find_all(['img', 'video', 'audio', 'script', 'link'])
    if not tags:
        return 1, "No external resources"
    external = sum(1 for t in tags for attr in ['src', 'href'] if t.get(attr) and urlparse(t.get(attr)).netloc and urlparse(t.get(attr)).netloc != urlparse(url).netloc)
    ratio = (external / len(tags)) * 100
    if ratio < 22:
        return 1, f"Low external resources: {ratio:.1f}%"
    elif 22 <= ratio <= 61:
        return 0, f"Moderate external resources: {ratio:.1f}%"
    else:
        return -1, f"High external resources: {ratio:.1f}%"

# 14. Suspicious Anchor - Check the ratio of external anchors to total anchors
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
    if ratio < 31:
        return 1, f"Low suspicious anchors: {ratio:.1f}%"
    elif 31 <= ratio <= 67:
        return 0, f"Moderate suspicious anchors: {ratio:.1f}%"
    else:
        return -1, f"High suspicious anchors: {ratio:.1f}%"

# 15. Links in Meta, Script, and Link Tags - Check the ratio of external links in these tags
def links_in_tags(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    tags = soup.find_all(['meta', 'script', 'link'])
    if not tags:
        return 1, "No such tags"
    external = sum(1 for t in tags for attr in ['src', 'href'] if t.get(attr) and urlparse(t.get(attr)).netloc and urlparse(t.get(attr)).netloc != urlparse(url).netloc)
    ratio = (external / len(tags)) * 100
    if ratio < 17:
        return 1, f"Legitimate: External links {ratio:.1f}%"
    elif 17 <= ratio <= 81:
        return 0, f"Suspicious: External links {ratio:.1f}%"
    else:
        return -1, f"Phishing: External links {ratio:.1f}%"

# 16. Server Form Handler - Check if the form action is empty, blank, or external
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

# 17. Submitting to Email - Check if the page contains a mailto: link
def submitting_to_email(content):
    if content is None:
        return -1, "Page inaccessible"
    if re.search(r'mail\(|mailto:', content, re.IGNORECASE):
        return -1, "Email submission detected"
    return 1, "No email submission"

# 18. Abnormal URL - Check if the URL structure is abnormal
def abnormal_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    extracted = tldextract.extract(url)
    registered_domain = extracted.registered_domain
    if is_ip_address(hostname) or hostname in [registered_domain, f"www.{registered_domain}"]:
        return 1, "URL matches domain"
    return -1, "Abnormal URL structure"

# HTML and JavaScript-Based Features
# 19. Redirects - Check the number of redirects
def website_forwarding(history):
    num_redirects = len(history)
    if num_redirects <= 1:
        return 1, f"{num_redirects} redirects (legitimate)"
    elif num_redirects >= 4:
        return -1, f"{num_redirects} redirects (phishing)"
    else:
        return 0, f"{num_redirects} redirects (suspicious)"

# 20. Status Bar Customization - Check if the status bar is modified
def status_bar_customization(content):
    if content is None:
        return -1, "Page inaccessible"
    if re.search(r'onmouseover\s*=\s*["\']window\.status\s*=', content, re.IGNORECASE):
        return -1, "Status bar modification detected"
    return 1, "No status bar changes"

# 21. Disabling Right Click - Check if right-click is disabled
def disabling_right_click(content):
    if content is None:
        return -1, "Page inaccessible"
    if 'event.button==2' in content.lower():
        return -1, "Right-click disabled"
    return 1, "Right-click enabled"

# 22. Using Popup Window - Check if the page uses popup windows
def using_popup_window(content):
    if content is None:
        return -1, "Page inaccessible"
    if 'window.open(' in content.lower():
        if re.search(r'<input[^>]*type=["\']?text["\']?', content, re.IGNORECASE):
            return -1, "Popup with text fields detected (phishing)"
        return 1, "Popup detected but no text fields (legitimate)"
    return 1, "No popups"

# 23. Using iFrames - Check if the page uses iframes
def iframe_redirection(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    iframes = soup.find_all('iframe')
    if not iframes:
        return 1, "No iframes"
    suspicious = sum(1 for i in iframes if i.get('frameborder', '').lower() == '0')
    return -1, f"{len(iframes)} iframes, {suspicious} suspicious" if suspicious else (1, f"{len(iframes)} iframes, legitimate")

# Domain-Based Features
# 24. Domain Age - Check the age of the domain
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

# 25. DNS Record - Check the number of DNS records
def dns_record(domain):
    if is_ip_address(domain):
        return 1, "IP address, DNS assumed"
    try:
        answers = dns.resolver.resolve(domain, 'A')
        return 1, f"{len(answers)} DNS records"
    except Exception:
        return -1, "No DNS records"

# 26. Website Traffic - Check the number of indexed pages
# def website_traffic(url):
#     """
#     Check website traffic by leveraging the google_index result.
#     Returns 1 if indexed (assumed traffic), -1 if not indexed (low traffic).
#     """
#     domain = urlparse(url).netloc
#     query = f"site:{domain}"
#
#     try:
#         results = list(search(query))
#         indexed_pages = len(results)
#
#         if indexed_pages > 100:
#             return 1, f"High traffic ({indexed_pages} pages indexed)"
#         elif indexed_pages > 10:
#             return 0, f"Moderate traffic ({indexed_pages} pages indexed)"
#         else:
#             return -1, "Low traffic (very few pages indexed)"
#
#     except Exception as e:
#         return -1, f"Google search failed: {str(e)}"


# 26. Website Traffic - Check the number of indexed pages
def website_traffic(url):
    """
    Check website traffic using Tranco rank.
    Returns 1 if rank < 100,000 (Legitimate), 0 if rank > 100,000 (Suspicious), -1 if no rank (Phishing).
    """
    domain = urlparse(url).netloc
    tranco_api_url = f"https://tranco-list.eu/api/ranks/domain/{domain}"

    try:
        response = requests.get(tranco_api_url)
        if response.status_code == 200:
            data = response.json()
            ranks = data.get('ranks', [])
            if ranks:
                latest_rank = ranks[-1].get('rank')
                if latest_rank is not None:
                    if latest_rank < 100000:
                        return 1, f"High traffic (Tranco rank: {latest_rank})"
                    else:
                        return 0, f"Moderate traffic (Tranco rank: {latest_rank})"
            return -1, "Low traffic (not recognized by Tranco)"
        elif response.status_code == 403:
            return -1, "Service temporarily unavailable"
        elif response.status_code == 429:
            return -1, "Rate limit exceeded"
        else:
            return -1, f"Unexpected status code: {response.status_code}"
    except Exception as e:
        return -1, f"Tranco rank check failed: {str(e)}"

# 27. PageRank - Check the Google PageRank
def page_rank(url):
    score, message = google_index(url)
    if score == 1:
        pagerank_value = calculate_pagerank(url)
        if pagerank_value < 0.2:
            return -1, f"Low PageRank ({pagerank_value})"
        return 1, f"High PageRank ({pagerank_value})"
    return -1, "Low authority (not indexed or error)"

def calculate_pagerank(url, damping_factor=0.85, iterations=100):
    try:
        params = {
            "engine": "google",
            "q": f"site:{url}",
            "api_key": "API_KEY"
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "organic_results" in data and data["organic_results"]:
            num_results = len(data["organic_results"])
            pagerank_value = (1 - damping_factor) + damping_factor * (num_results / 100.0)
            return pagerank_value
        return 0.0
    except Exception as e:
        print(f"Error retrieving PageRank value: {e}")
        return 0.0

# 28. Google Index - Check if the page is indexed by Google
def google_index(url):
    try:
        query = f"site:{urlparse(url).netloc}"
        results = search(query)
        indexed = any(results)
        return (1, "Indexed by Google") if indexed else (-1, "Not indexed")
    except Exception as e:
        return -1, f"Google index check failed: {e}"

# 29. Number of Links Pointing - Check the number of outbound external links
def number_of_links_pointing(url, soup):
    if soup is None:
        return -1, "Page inaccessible"
    links = soup.find_all('a', href=True)
    external = sum(1 for a in links if urlparse(a['href']).netloc and urlparse(a['href']).netloc != urlparse(url).netloc)
    if external == 0:
        return -1, "No outbound external links (Phishing)"
    elif 0 < external <= 2:
        return 0, f"{external} outbound external links (Suspicious)"
    else:
        return 1, f"{external} outbound external links (Legitimate)"

# 30. Phishing Reports - Check the number of phishing reports
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
    # for feature, (score, reason) in features.items():
    #     status = "SAFE" if score == 1 else "WARNING" if score == 0 else "DANGER"
    #     print(f"{feature:25} [{status:^7}] {reason} (Score: {score})")

    scores = []
    for feature, (score, reason) in features.items():
        scores.append(score)

    return scores

if __name__ == "__main__":
    test_url = "https://www.apple.com"
    run_checks(test_url)