function getDomain(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get({ blockedSites: [] }, (data) => {
    updateBlockedSites(data.blockedSites);
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "updateBlockedSites") {
    updateBlockedSites(message.sites);
  }
});

function updateBlockedSites(blockedSites) {
  chrome.declarativeNetRequest.getDynamicRules((existingRules) => {
    const ruleIdsToRemove = existingRules.map((rule) => rule.id);

    chrome.declarativeNetRequest.updateDynamicRules(
      { removeRuleIds: ruleIdsToRemove },
      () => {
        const newRules = blockedSites.map((site, index) => ({
          id: index + 1, // Unique rule ID
          priority: 1,
          action: { type: "block" },
          condition: {
            urlFilter: `*://${site}/*`,
            resourceTypes: ["main_frame"],
          },
        }));

        chrome.declarativeNetRequest.updateDynamicRules(
          { addRules: newRules },
          () => {
            console.log("Updated blocked sites:", blockedSites);
          }
        );
      }
    );
  });
}

chrome.storage.onChanged.addListener((changes) => {
  if (changes.blockedSites) {
    updateBlockedSites(changes.blockedSites.newValue);
  }
});
