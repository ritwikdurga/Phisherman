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

# Disable warnings for unverified HTTPS requests
requests.packages.urllib3.disable_warnings()


# Helper functions
def get_domain_info(url):
    return tldextract.extract(url)


def get_page_content(url):
    try:
        response = requests.get(url, timeout=10, verify=False)
        return response.text, response.history
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return None, []


# Address Bar based Features
def having_ip_address(url):
    match = re.search(r'\d+\.\d+\.\d+\.\d+', url)
    return (-1, "IP addresses in URLs are often used by phishers") if match else (1, "No IP address detected")


def url_length(url):
    length = len(url)
    if length < 54:
        return 1, "Short URL (safe)"
    elif 54 <= length <= 75:
        return 0, "Medium URL (suspicious)"
    else:
        return -1, "Long URL (phishing)"


def shortening_service(url):
    shorteners = r'(bit\.ly|goo\.gl|shorte\.st|tinyurl|tr\.im|ow\.ly|is\.gd|cli\.gs|' \
                 r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|' \
                 r'snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|' \
                 r'snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|' \
                 r'om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|qr\.ae|adf\.ly|goo\.gl|' \
                 r'bitly\.com|cur\.lv|tinyurl\.com|tr\.im|ow\.ly|bit\.ly|ity\.im|q\.gs|' \
                 r'is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|' \
                 r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|' \
                 r'tweez\.me|v\.gd|tr\.im|link\.zip\.net)'
    return (-1, "URL shortening service detected") if re.search(shorteners, url) else (1, "No URL shortening")


def having_at_symbol(url):
    return (-1, "@ symbol in URL (phishing)") if '@' in url else (1, "No @ symbol")


def double_slash_redirecting(url):
    return (-1, "Double slash redirection") if url.rfind('//') > 6 else (1, "No double slash redirection")


def prefix_suffix(url):
    domain = get_domain_info(url).domain
    return (-1, "Hyphen in domain name") if '-' in domain else (1, "No hyphens in domain")


def sub_domains(url):
    # Use tldextract to reliably extract parts of the domain.
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain  # e.g., "www" for "www.google.com"
    # Count only non-empty subdomains (split by ".")
    subdomain_count = len([s for s in subdomain.split('.') if s])
    # A single "www" is common and safe.
    if subdomain_count == 0 or (subdomain_count == 1 and subdomain.lower() == "www"):
        return 1, "Single (or no) subdomain"
    elif subdomain_count == 1:
        return 1, "Single subdomain"
    elif subdomain_count == 2:
        return 0, "Two subdomains (suspicious)"
    else:
        return -1, "Multiple subdomains (phishing)"


def https_certificate(url):
    try:
        # Parse hostname and port
        parsed = urlparse(url)
        hostname = parsed.hostname
        port = parsed.port if parsed.port else 443

        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        cert_date_str = cert.get('notAfter')
        if not cert_date_str:
            return -1, "Certificate does not have an expiration date"
        # Example format: 'Dec  8 12:00:00 2025 GMT'
        cert_date = datetime.strptime(cert_date_str, '%b %d %H:%M:%S %Y %Z')
        # Make sure cert_date is timezone-aware (assuming it's in UTC)
        cert_date = cert_date.replace(tzinfo=timezone.utc)
        days_valid = (cert_date - datetime.now(timezone.utc)).days
        if days_valid >= 365:
            return 1, f"Valid SSL certificate (>1 year, expires in {days_valid} days)"
        else:
            return -1, f"Short-term SSL certificate (expires in {days_valid} days)"
    except Exception as e:
        return -1, f"Error retrieving certificate details: {e}"


def non_standard_port(url):
    parsed = urlparse(url)
    if parsed.port:
        return (-1, f"Non-standard port {parsed.port}") if parsed.port not in [80, 443] else (1, "Standard port")
    return 1, "Default port"


def https_domain_token(url):
    domain = get_domain_info(url).domain
    return (-1, "HTTPS in domain name") if 'https' in domain.lower() else (1, "No HTTPS in domain")


# Abnormal based Features
def external_request_ratio(url):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        tags = soup.find_all(['img', 'script', 'link'])
        total = len(tags)
        external = 0
        for t in tags:
            attr = 'src' if t.name == 'img' else 'href'
            link = t.get(attr, '')
            if link:
                if urlparse(link).netloc and urlparse(link).netloc != urlparse(url).netloc:
                    external += 1
        ratio = (external / total) * 100 if total > 0 else 0
        if ratio < 22:
            return 1, "Low external requests"
        elif 22 <= ratio <= 61:
            return 0, "Moderate external requests"
        else:
            return -1, "High external requests"
    except Exception as e:
        return -1, f"Error checking external requests: {e}"


def anchor_url_analysis(url):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        anchors = soup.find_all('a')
        external = sum(
            1 for a in anchors if a.get('href') and urlparse(a.get('href')).netloc and
            urlparse(a.get('href')).netloc != urlparse(url).netloc
        )
        ratio = (external / len(anchors)) * 100 if anchors else 0
        if ratio < 31:
            return 1, "Low external anchors"
        elif 31 <= ratio <= 67:
            return 0, "Moderate external anchors"
        else:
            return -1, "High external anchors"
    except Exception as e:
        return -1, f"Error checking anchor URLs: {e}"


def meta_script_link_analysis(url):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        tags = soup.find_all(['meta', 'script', 'link'])
        external = 0
        for t in tags:
            attr = t.get('src') or t.get('href', '')
            if attr:
                if urlparse(attr).netloc and urlparse(attr).netloc != urlparse(url).netloc:
                    external += 1
        ratio = (external / len(tags)) * 100 if tags else 0
        return (-1, f"High external resources: {ratio:.1f}%") if ratio > 50 else (1, f"External resources: {ratio:.1f}%")
    except Exception as e:
        return -1, f"Error checking meta/script/link tags: {e}"


def server_form_handler(url):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        forms = soup.find_all('form')
        for form in forms:
            action = form.get('action', '')
            if not action or action.lower() == 'about:blank':
                return -1, "Blank form action"
            if urlparse(action).netloc and urlparse(action).netloc != urlparse(url).netloc:
                return 0, "External form handler"
        return 1, "Local form handlers"
    except Exception as e:
        return -1, f"Error checking forms: {e}"


def email_submission(url):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        if soup.find(string=re.compile(r'mail\(|mailto:')):
            return -1, "Email submission detected"
        return 1, "No email submission"
    except Exception as e:
        return -1, f"Error checking email submission: {e}"


def abnormal_url(url):
    parsed = urlparse(url)
    host = parsed.hostname or ""
    extracted = tldextract.extract(url)
    registered_domain = extracted.registered_domain
    # Allow for "www." prefix as acceptable.
    if host == registered_domain or host == f"www.{registered_domain}":
        return (1, "Hostname matches registered domain")
    else:
        return (-1, "Hostname does not match registered domain")



# HTML/JavaScript based Features
def redirect_count(history):
    return (-1, f"{len(history)} redirects") if len(history) > 2 else (1, "Normal redirects")


def status_bar_modification(url):
    try:
        content = requests.get(url, timeout=10).text.lower()
        if 'onmouseover' in content:
            return -1, "Status bar modification detected"
        return 1, "No status bar changes"
    except Exception as e:
        return -1, f"Error checking status bar: {e}"


def right_click_disable(url):
    try:
        content = requests.get(url, timeout=10).text.lower()
        if 'event.button === 2' in content:
            return -1, "Right-click disabled"
        return 1, "Right-click enabled"
    except Exception as e:
        return -1, f"Error checking right-click: {e}"


def popup_windows(url):
    try:
        content = requests.get(url, timeout=10).text.lower()
        if 'window.open(' in content:
            return -1, "Popup windows detected"
        return 1, "No popups"
    except Exception as e:
        return -1, f"Error checking popups: {e}"


def iframe_usage(url):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        iframes = soup.find_all('iframe')
        return (-1, f"{len(iframes)} iframes detected") if iframes else (1, "No iframes")
    except Exception as e:
        return -1, f"Error checking iframes: {e}"


# Domain based Features
def domain_age(domain):
    try:
        w = whois.whois(domain)
        if w.creation_date:
            # Handle if creation_date is a list
            if isinstance(w.creation_date, list):
                creation_date = min(w.creation_date)
            else:
                creation_date = w.creation_date
            # If creation_date is timezone-aware, convert to naive for subtraction.
            if creation_date.tzinfo is not None:
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            age = (datetime.now() - creation_date).days
            return (1, f"Domain age: {age} days") if age >= 180 else (-1, "New domain")
        return -1, "No creation date"
    except Exception as e:
        return -1, f"WHOIS lookup failed: {e}"


def dns_records(domain):
    try:
        resolver = dns.resolver.Resolver()
        answers = resolver.resolve(domain, 'A')
        return 1, f"{len(answers)} DNS records"
    except Exception as e:
        return -1, f"No DNS records: {e}"


def website_traffic(url):
    # Placeholder for actual traffic API integration
    return 0, "Traffic check not implemented"

# Add api key here
def google_pagerank(url):
    try:
        params = {
            "engine": "google",
            "q": f"site:{url}",
            "api_key": "key"
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "organic_results" in data and data["organic_results"]:
            return 1, "Indexed by Google"
        else:
            return -1, "Not indexed"
    except Exception as e:
        return -1, f"Google PageRank check failed: {e}"


def google_index(url):
    try:
        query = f"site:{url}"
        results = list(search(query, num_results=1))
        return (1, "Indexed by Google") if results else (-1, "Not indexed")
    except Exception as e:
        return -1, f"Google check failed: {e}"


def external_links(url):
    try:
        soup = BeautifulSoup(requests.get(url, timeout=10).content, 'html.parser')
        links = sum(
            1 for a in soup.find_all('a')
            if a.get('href') and urlparse(a.get('href')).netloc and
            urlparse(a.get('href')).netloc != urlparse(url).netloc
        )
        return (-1, "No external links") if links == 0 else (1, f"{links} external links")
    except Exception as e:
        return -1, f"Error checking external links: {e}"


def phishing_reports(domain):
    # Placeholder for PhishTank API integration
    return 0, "Phishing reports check not implemented"


def run_checks(url):
    domain_info = get_domain_info(url)
    content, history = get_page_content(url)

    features = {
        # Address Bar Features
        'IP Address': having_ip_address(url),
        'URL Length': url_length(url),
        'Shortening Service': shortening_service(url),
        '@ Symbol': having_at_symbol(url),
        'Double Slash': double_slash_redirecting(url),
        'Prefix/Suffix': prefix_suffix(url),
        'Sub Domains': sub_domains(url),
        'HTTPS Certificate': https_certificate(url),
        'Non-Standard Port': non_standard_port(url),
        'HTTPS in Domain': https_domain_token(url),

        # Abnormal Features
        'External Requests': external_request_ratio(url),
        'Anchor Links': anchor_url_analysis(url),
        'Meta/Script/Links': meta_script_link_analysis(url),
        'Form Handling': server_form_handler(url),
        'Email Submission': email_submission(url),
        'Abnormal URL': abnormal_url(url),

        # HTML/JS Features
        'Redirects': redirect_count(history),
        'Status Bar': status_bar_modification(url),
        'Right Click': right_click_disable(url),
        'Popups': popup_windows(url),
        'Iframes': iframe_usage(url),

        # Domain Features
        'Domain Age': domain_age(domain_info.registered_domain),
        'DNS Records': dns_records(domain_info.registered_domain),
        'Website Traffic': website_traffic(url),
        'PageRank': google_pagerank(url),
        'Google Index': google_index(url),
        'External Links': external_links(url),
        'Phishing Reports': phishing_reports(domain_info.registered_domain)
    }

    print(f"\nSecurity Analysis for: {url}\n{'=' * 40}")
    for feature, (score, reason) in features.items():
        status = "SAFE" if score == 1 else "WARNING" if score == 0 else "DANGER"
        print(f"{feature:20} [{status:^7}] {reason}")


if __name__ == "__main__":
    # Uncomment these lines to use interactive input:
    # url = input("Enter URL to analyze: ").strip()
    # run_checks(url)

    # Hard-coded URL for testing:
    url = "https://www.google.com"
    run_checks(url)
