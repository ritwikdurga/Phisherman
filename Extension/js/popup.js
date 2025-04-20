document.addEventListener("DOMContentLoaded", function () {
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const urlInput = document.getElementById("urlInput");
  const goButton = document.getElementById("goButton");
  const clearButton = document.getElementById("clearButton");
  const responseSection = document.getElementById("response");
  const closePopup = document.getElementById("closePopup");

  let currentUrl = "";

  // Function to update the theme icon
  function updateThemeIcon() {
    const isDarkMode = document.body.classList.contains("dark-mode");
    themeIcon.src = isDarkMode
      ? "../resources/dark_mode_img.jpg"
      : "../resources/light_mode_img.png";
  }

  // Check for saved theme preference
  if (localStorage.getItem("theme") === "dark") {
    document.body.classList.add("dark-mode");
  }

  // Toggle theme on button click
  themeToggle.addEventListener("click", function () {
    document.body.classList.toggle("dark-mode");
    const isDark = document.body.classList.contains("dark-mode");
    localStorage.setItem("theme", isDark ? "dark" : "light");
    updateThemeIcon();
  });

  updateThemeIcon(); // Set initial state of the theme icon

  closePopup.addEventListener("click", function () {
    window.close(); // Close the popup
  });

  // Retrieve the current URL
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (tabs.length === 0 || !tabs[0].url) return;

    currentUrl = tabs[0].url;
    const invalidUrls = ["chrome://newtab/", "about:newtab", "edge://newtab/"];

    if (!invalidUrls.includes(currentUrl)) {
      urlInput.value = currentUrl;
    }
  });

  // Function to check if a site is blocked
  function isSiteBlocked(url, callback) {
    chrome.storage.local.get({ blockedSites: [] }, (data) => {
      const blockedSites = data.blockedSites;
      const domain = new URL(url).hostname;
      callback(blockedSites.includes(domain));
    });
  }

  // Function to validate the URL
  function isValidUrl(url) {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  // Handle "Go" button click to analyze the URL
  goButton.addEventListener("click", function () {
    const url = urlInput.value.trim();
    responseSection.innerHTML = "";

    // Check if the URL is valid before proceeding
    if (!isValidUrl(url)) {
      responseSection.innerHTML = `<p style="color: #d32f2f; font-weight: bold;">❌ Please enter a valid URL.</p>`;
      return;
    }

    // Display "Checking URL..." message
    responseSection.innerHTML = `
      <div style="max-width: 400px; padding: 20px; border-radius: 10px; text-align: center; background-color: #e0f7fa;">
        <p style="color: #18181b; font-weight: bold;">🔍 Checking URL...</p>
      </div>
    `;

    // Initiate API call
    fetch("http://localhost:5050/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: url }),
    })
      .then((res) => res.json())
      .then((data) => {
        const isPhishy = !data.prediction;
        console.log(data);

        const formattedUrl = new URL(url).href;
        const domain = new URL(url).hostname;
        const isSameUrl = formattedUrl === currentUrl;

        isSiteBlocked(url, (isBlocked) => {
          if (isBlocked) {
            responseSection.innerHTML = `
              <div style="max-width: 400px; padding: 20px; border-radius: 10px; text-align: center; background-color: #ffebee;">
                <h2 style="color: #d32f2f;">❌ This site is blocked!</h2>
                <p style="color: #333;">${domain} is in your blocked list.</p>
              </div>
            `;
            return;
          }

          responseSection.innerHTML = `
            <div style="max-width: 400px; padding: 20px; border-radius: 10px; text-align: center; background-color: ${
              isPhishy ? "#ffebee" : "#e8f5e9"
            };">
              <h2 class="${isPhishy ? "phishy-text" : "safe-text"}">
                ${
                  isPhishy ? "⚠️ Phishy URL Detected!" : "✅ Safe URL Detected!"
                }
              </h2>
              <p class="${isPhishy ? "phishy-text" : "safe-text"}" >
                ${
                  isSameUrl
                    ? isPhishy
                      ? "The URL you are currently in is potentially dangerous. Do not proceed."
                      : "The URL you are currently in is safe."
                    : isPhishy
                    ? "The URL you entered is potentially dangerous. Do not proceed."
                    : "The URL you entered seems safe."
                }
              </p>
              ${
                !isSameUrl
                  ? `
                  <div style="margin-top: 15px;">
                    <a href="${formattedUrl}" target="_blank" rel="noopener noreferrer" style="
                      display: inline-block;
                      padding: 10px 15px;
                      border-radius: 5px;
                      text-decoration: none;
                      font-weight: bold;
                      color: white;
                      background-color: ${isPhishy ? "#d32f2f" : "#2e7d32"};
                      font-family: 'Inter', sans-serif;
                    ">
                      ${isPhishy ? "⚠️ Proceed Anyway" : "Open Website"}
                    </a>
                  </div>
                `
                  : ""
              }
              ${
                !isBlocked
                  ? `
                  <div style="margin-top: 10px;">
                    <a href="#" id="blockLink" class="block-link" style="color: ${
                      isPhishy ? "#d32f2f" : "#2e7d32"
                    };">
                      Block this website
                    </a>
                  </div>
                `
                  : ""
              }
              <p class="disclaimer-text" style="margin-top: 30px; font-size: 12px; color: #757575;">
                This is an AI-generated result and may not be 100% accurate. Be cautious when opening links.
              </p>
            </div>
          `;

          if (!isBlocked) {
            document
              .getElementById("blockLink")
              .addEventListener("click", function (e) {
                e.preventDefault();
                chrome.storage.local.get({ blockedSites: [] }, (data) => {
                  if (!data.blockedSites.includes(domain)) {
                    const updatedSites = [...data.blockedSites, domain];
                    chrome.storage.local.set(
                      { blockedSites: updatedSites },
                      () => {
                        chrome.runtime.sendMessage({
                          action: "updateBlockedSites",
                          sites: updatedSites,
                        });
                        responseSection.innerHTML = `<p style="color: #2e7d32;">✅ ${domain} has been blocked.</p>`;
                      }
                    );
                  }
                });
              });
          }
        });
      })
      .catch((error) => {
        console.error("Error:", error);
        responseSection.innerHTML = `<p style="color: #d32f2f; font-weight: bold;">❌ An error occurred while checking the URL.</p>`;
      });
  });

  // Handle "Clear" button click
  clearButton.addEventListener("click", function () {
    urlInput.value = "";
    responseSection.innerHTML = "";
  });
});
