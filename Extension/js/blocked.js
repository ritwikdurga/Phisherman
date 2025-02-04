document.addEventListener("DOMContentLoaded", function () {
  const blockedList = document.getElementById("blockedList");

  function refreshBlockedList() {
    chrome.storage.local.get({ blockedSites: [] }, (data) => {
      blockedList.innerHTML = "";

      if (data.blockedSites.length === 0) {
        const noBlockedMessage = document.createElement("p");
        noBlockedMessage.textContent = "No blocked websites.";
        noBlockedMessage.classList.add("no-blocked"); // Apply new styling
        blockedList.appendChild(noBlockedMessage);
        return;
      }

      data.blockedSites.forEach((site) => {
        const li = document.createElement("li");
        li.classList.add("blocked-item");

        const domainInfo = document.createElement("div");
        domainInfo.classList.add("domain-info");

        const favicon = document.createElement("img");
        favicon.classList.add("domain-icon");
        favicon.src = `https://www.google.com/s2/favicons?sz=32&domain=${site}`;
        favicon.alt = "Favicon";

        const domainName = document.createElement("span");
        domainName.classList.add("domain-name");
        domainName.textContent = site;

        domainInfo.appendChild(favicon);
        domainInfo.appendChild(domainName);

        const unblockButton = document.createElement("button");
        unblockButton.classList.add("unblock-button");
        unblockButton.textContent = "Unblock";
        unblockButton.addEventListener("click", () => {
          unblockSite(site);
        });

        li.appendChild(domainInfo);
        li.appendChild(unblockButton);
        blockedList.appendChild(li);
      });
    });
  }

  function unblockSite(site) {
    chrome.storage.local.get({ blockedSites: [] }, (data) => {
      const updatedSites = data.blockedSites.filter((s) => s !== site);
      chrome.storage.local.set({ blockedSites: updatedSites }, () => {
        refreshBlockedList();
        chrome.runtime.sendMessage({
          action: "updateBlockedSites",
          sites: updatedSites,
        });
      });
    });
  }

  chrome.storage.onChanged.addListener((changes) => {
    if (changes.blockedSites) {
      refreshBlockedList();
    }
  });

  refreshBlockedList();
});
