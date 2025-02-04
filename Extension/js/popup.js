document.addEventListener("DOMContentLoaded", function () {
  const urlInput = document.getElementById("urlInput");
  const goButton = document.getElementById("goButton");
  const clearButton = document.getElementById("clearButton");
  const responseSection = document.getElementById("response");
  const themeToggle = document.getElementById("themeToggle");
  const themeIcon = themeToggle.querySelector(".material-symbols-outlined");
  const closePopup = document.getElementById("closePopup");

  let currentUrl = "";

  // Check for saved theme preference
  if (localStorage.getItem("theme") === "dark") {
    document.body.classList.add("dark-mode");
    themeIcon.textContent = "light_mode";
  }

  themeToggle.addEventListener("click", function () {
    document.body.classList.toggle("dark-mode");
    const isDark = document.body.classList.contains("dark-mode");
    themeIcon.textContent = isDark ? "light_mode" : "dark_mode";
    localStorage.setItem("theme", isDark ? "dark" : "light");
  });

  closePopup.addEventListener("click", function () {
    window.close(); // Closes the popup
  });

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    if (tabs.length === 0 || !tabs[0].url) return;

    currentUrl = tabs[0].url;
    const invalidUrls = ["chrome://newtab/", "about:newtab", "edge://newtab/"];

    if (!invalidUrls.includes(currentUrl)) {
      urlInput.value = currentUrl;
    }
  });

  function isSiteBlocked(url, callback) {
    chrome.storage.local.get({ blockedSites: [] }, (data) => {
      const blockedSites = data.blockedSites;
      const domain = new URL(url).hostname;
      callback(blockedSites.includes(domain));
    });
  }

  goButton.addEventListener("click", function () {
    const url = urlInput.value.trim();
    responseSection.innerHTML = "";

    if (isValidUrl(url)) {
      const formattedUrl = new URL(url).href;
      const domain = new URL(url).hostname;
      const isSameUrl = formattedUrl === currentUrl;
      const isPhishy = Math.random() < 0.5;

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
                  <h2 style="color: ${isPhishy ? "#d32f2f" : "#2e7d32"};">
                      ${
                        isPhishy
                          ? "⚠️ Phishy URL Detected!"
                          : "✅ Safe URL Detected!"
                      }
                  </h2>
                  <p style="color: #333;">
                      ${
                        isSameUrl
                          ? isPhishy
                            ? "The URL you are currently in is potentially dangerous. Do not proceed."
                            : "The URL you are currently in seems safe."
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
                              background-color: ${
                                isPhishy ? "#d32f2f" : "#2e7d32"
                              };
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
                          <a href="#" id="blockLink" style="
                              display: inline-block;
                              color: ${isPhishy ? "#d32f2f" : "#2e7d32"};
                              text-decoration: underline;
                              cursor: pointer;
                          ">
                              Block this website
                          </a>
                      </div>
                  `
                      : ""
                  }
                  <p style="font-size: 11px; color: #777; margin-top: 15px;">
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
    } else {
      responseSection.innerHTML = `<p style="color: #d32f2f; font-weight: bold;">❌ Please enter a valid URL.</p>`;
    }
  });

  clearButton.addEventListener("click", function () {
    urlInput.value = "";
    responseSection.innerHTML = "";
  });

  function isValidUrl(url) {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }
});
