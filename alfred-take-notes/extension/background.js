let currentSidePanelPort = null;
let lastKnownVideoMetadata = null;

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === "intelligence-sidepanel-pipe") {
    currentSidePanelPort = port;
    if (lastKnownVideoMetadata) {
      currentSidePanelPort.postMessage({ type: "CONTEXT_UPDATE", data: lastKnownVideoMetadata });
    } else {
      evaluatedActiveContextChange();
    }
    port.onDisconnect.addListener(() => {
      currentSidePanelPort = null;
    });
  }
});

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "DYNAMIC_VIDEO_CHANGE") {
    lastKnownVideoMetadata = message.data;
    if (currentSidePanelPort) {
      currentSidePanelPort.postMessage({ type: "CONTEXT_UPDATE", data: message.data });
    }
  }
});

async function ensureContentScriptInjected(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { action: "ping" });
  } catch (err) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ["content.js"]
      });
      await new Promise(resolve => setTimeout(resolve, 200));
    } catch (e) {
      console.error("Programmatic injection block:", e);
    }
  }
}

async function evaluatedActiveContextChange() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Exact structural RegExp validation
    const isYouTubeWatchPage = tab && tab.url && /^https:\/\/(www\.)?youtube\.com\/watch/.test(tab.url);

    if (!isYouTubeWatchPage) {
      lastKnownVideoMetadata = null;
      if (currentSidePanelPort) {
        currentSidePanelPort.postMessage({ type: "CONTEXT_OFFLINE" });
      }
      return;
    }

    await ensureContentScriptInjected(tab.id);

    chrome.tabs.sendMessage(tab.id, { action: "fetchActiveVideoDOM" }, (response) => {
      if (!chrome.runtime.lastError && response) {
        lastKnownVideoMetadata = response;
        if (currentSidePanelPort) {
          currentSidePanelPort.postMessage({ type: "CONTEXT_UPDATE", data: response });
        }
      }
    });
  } catch (err) {
    console.error("Context engine tracking warning:", err);
  }
}

chrome.tabs.onActivated.addListener(() => evaluatedActiveContextChange());
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'complete' || changeInfo.url) {
    evaluatedActiveContextChange();
  }
});
chrome.windows.onFocusChanged.addListener(() => evaluatedActiveContextChange());
