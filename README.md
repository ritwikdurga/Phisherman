<h1>
  <img src="/Extension/resources/images/dark_mode_img.jpg" alt="Logo" width="50" style="position: relative; top: 20px; margin-right: 10px;">
  Phisherman - Secure Browsing Made Easy
</h1>


Phisherman is a browser extension designed to protect users from phishing attacks by analyzing, detecting, and blocking suspicious URLs in real-time. Leveraging a machine learning-powered backend and third-party APIs, it ensures safer web navigation through both automatic and manual URL analysis.



##  Table Of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Run Locally](#run-locally)
- [Tech Stack](#tech-stack)
- [API Reference](#api-reference)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Authors](#authors)
- [FAQ](#faq)

## About

Phisherman enhances online safety by detecting phishing threats through machine learning models and external APIs. It utilizes a variant of the **Extreme Learning Machine (ELM)** model stored as pickle file to classify URLs as legitimate or phishing. The extension connects to a Flask-based backend and also uses **SerpAPI** for page ranking and **VirusTotal** for malicious content checks during the feature extraction process.

Users can:
- Automatically scan current tab URLs.
- Manually input and verify any URL.
- Block access to flagged phishing sites.

## Features

- Automatically detects URLs and allows you to analyze URLs as you browse or allows manual URL input for checking.
- Uses a variation of ELM model to classify URLs as safe or phishing based on extracted features.
- Provides an option to block suspicious or confirmed phishing URLs to prevent access.
- Utilises SerpAPI for page rank calculation and VirusTotal for malicious content detection.
- Offers a popup interface for easy URL checking and blocking, with light and dark mode support.
- Designed to work with modern browsers supporting WebExtensions.

## Installation

Before you can start using this project, you need to set up your environment. Follow these steps for installation and configuration:

1. **Install Python:**
   
   If you don't already have Python installed on your system, you can download it from the official Python website:

   - [Python Official Website](https://www.python.org/downloads/)

   Please choose the appropriate version (recommended: Python 3.12.0) for your operating system.

2. **Clone the Repository**:

   Clone the Phisherman repository to your local machine:

      ```bash
      git clone https://github.com/ritwikdurga/Phisherman.git
      ```

3. **Install Project Dependencies:**

   - Open your terminal or command prompt.
   - Navigate to the project directory using the `cd` command.
   - Run the following command to install the required Python libraries from the provided `requirements.txt` file:

      ```shell
      pip install -r requirements.txt
      ```

4. **Set Up API Keys**:

   - Create a `.env` file in the `server` directory.
   - Add the following API keys:

     ```
     SERPAPI_KEY=your_serpapi_key
     VT_API_KEY=your_virustotal_key
     ```
   - Obtain keys from SerpAPI and VirusTotal.

5. **Load the Browser Extension**:

   - Open your browser's extension management page (e.g., `chrome://extensions/` for Chrome).
   - Enable "Developer mode" and select "Load unpacked."
   - Choose the `Extension` folder from the Phisherman project directory.

6. **Start the Backend Server**:

   - Navigate to the `server` directory:

     ```bash
     cd server
     ```
   - Run the Flask server:

     ```bash
     python app.py
     ```
   - The server will start on `http://localhost:5050` (ensure the port matches the extension’s configuration).


## Dependencies

- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [seaborn](https://github.com/mwaskom/seaborn)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [python-whois](https://github.com/richardpenman/whois)
- [requests](https://github.com/psf/requests)
- [googlesearch-python](https://github.com/Nv7-GitHub/googlesearch)
- [beautifulsoup4](https://github.com/wention/BeautifulSoup4)
- [tldextract](https://github.com/john-kurkowski/tldextract)
- [dnspython](https://github.com/rthalley/dnspython)
- [flask](https://github.com/pallets/flask)
- [flask-cors](https://github.com/corydolphin/flask-cors)
- [mlxtend](https://github.com/rasbt/mlxtend)
- [joblib](https://github.com/joblib/joblib)


## Run Locally

To run Phisherman locally, follow these steps:

1. **Start the Backend Server**:

   - Navigate to the `server` directory:

     ```bash
     cd server
     ```
   - Run the Flask server:

     ```bash
     python app.py
     ```
   - The server will start on `http://localhost:5050` (ensure the port matches the extension’s configuration).

2. **Load the Extension**:

   - Ensure the extension is loaded in your browser as described in the Installation section.

3. **Test the Extension**:

   - Open the browser, click the Phisherman extension icon, and input a URL to analyze or let it automatically detect URLs.
   - Use the block feature for suspicious URLs.

## Tech Stack

- **Frontend**:
  - **HTML/CSS/JavaScript** for the extension’s popup and blocked page.
  - Custom fonts (Shadow Hand Font) and icons for UI.
- **Backend**:
  - **Flask (Python)** for the server handling URL analysis.
  - Variation of ELM model for phishing detection.
- **APIs**:
  - **SerpAPI** for page rank calculation.
  - **VirusTotal** for malicious content detection.
- **Data**:
  - ARFF and CSV datasets for training the phishing detection model.


## Screenshots

![Screenshot 1](/Extension/resources/images/Picture1.png)
![Screenshot 2](/Extension/resources/images/Picture2.png)
![Screenshot 3](/Extension/resources/images/Picture3.png)
![Screenshot 4](/Extension/resources/images/Picture4.png)


## API Reference

### SerpAPI

#### Calculate PageRank

```http
GET https://serpapi.com/search
```

| Parameter | Type     | Description                                 |
| --------- | -------- | ------------------------------------------- |
| `engine`  | `string` | **Required**. Set to `google`.              |
| `q`       | `string` | **Required**. Query in `site:{url}` format. |
| `api_key` | `string` | **Required**. Your SerpAPI key.             |

#### Response:

```json
{
  "organic_results": [
    {
      "position": 1,
      "title": "Sample Result",
      ...
    }
  ]
}
```

### VirusTotal API

#### Check URL for Malicious Content

```http
POST https://www.virustotal.com/api/v3/urls
```

| Parameter | Type     | Description                   |
| --------- | -------- | ----------------------------- |
| `url`     | `string` | **Required**. URL to analyze. |

| Header         | Value                               |
| -------------- | ----------------------------------- |
| `x-apikey`     | Your VirusTotal API key             |
| `accept`       | `application/json`                  |
| `content-type` | `application/x-www-form-urlencoded` |

#### Response:

```json
{
  "data": {
    "id": "analysis_id",
    "attributes": {
      "status": "completed",
      "stats": {
        "malicious": 0,
        "suspicious": 0,
        "harmless": 72
      }
    }
  }
}
```

## Usage

**Automatic URL Detection**:

   - Browse the web, and Phisherman will automatically capture the URL of the active tab. You can click "Check" to analyze that particular URL you are currently on.
   - The popup will display whether the URL is safe or phishing.

**Manual URL Check**:

   - Open the Phisherman extension popup.
   - Enter a URL (e.g., `http://example.com`) and click "Check"
   - View the result (e.g., "Safe" or "Phishing").

**Blocking a URL**:

   - If a URL is flagged as phishing or seems suspicious, click "Block URL" in the popup.
   - The extension will prevent access to the URL until unblocked again from the list of blocked URLs.

## Authors

- [**ritwikdurga**](https://www.github.com/ritwikdurga)
- [**Nandu-25**](https://www.github.com/Nandu-25)
- [**sudeepreddy999**](https://github.com/sudeepreddy999)

## License

This project is licensed under the [MIT License](./LICENSE). See the **LICENSE** file for details.

## FAQ

#### How does Phisherman detect phishing URLs?

A: Phisherman uses a variation of the ELM model trained on phishing website datasets to analyze URLs. Think of it as a digital Sherlock Holmes, sniffing out suspicious links and declaring, "Elementary, my dear Watson!"

#### Can I use Phisherman without API keys?

A: Technically, yes, but it would be like driving a car without wheels—possible, but not very effective. API keys for SerpAPI and VirusTotal unlock the full potential of Phisherman, and the free tiers are generous enough to get you started.

#### Is Phisherman compatible with all browsers?

A: Phisherman works with all Chromium-based browsers like Chrome, Edge, and even Firefox. If your browser supports WebExtensions, you're good to go. If not, maybe it's time to upgrade from that ancient relic you're using!

#### Does Phisherman slow down my browsing?

A: Not at all! Phisherman is as lightweight as a feather. It works quietly in the background, ensuring your browsing experience remains smooth while it hunts down phishing threats like a ninja.

## References

For further reference related to the dataset or research, refer to the [Phishing Websites](./ML/phishing_websites/) folder.
