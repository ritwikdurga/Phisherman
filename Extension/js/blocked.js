document.addEventListener("DOMContentLoaded", function () {
  const blockedList = document.getElementById("blockedList");

  function refreshBlockedList() {
    chrome.storage.local.get({ blockedSites: [] }, (data) => {
      blockedList.innerHTML = "";
      data.blockedSites.forEach((site) => {
        const li = document.createElement("li");
        li.textContent = site;

        const unblockButton = document.createElement("button");
        unblockButton.textContent = "Unblock";
        unblockButton.style.marginLeft = "10px";
        unblockButton.addEventListener("click", () => {
          unblockSite(site);
        });

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
