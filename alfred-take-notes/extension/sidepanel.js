const BACKEND_ENDPOINT = "http://localhost:3000/api";
let activeVideoMetadata = null;

async function forceSyncActiveTabContext() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const isYouTubeWatchPage = tab && tab.url && /^https:\/\/(www\.)?youtube\.com\/watch/.test(tab.url);
    
    if (!isYouTubeWatchPage) {
      renderOfflineState();
      return;
    }

    try {
      await chrome.tabs.sendMessage(tab.id, { action: "ping" });
    } catch (pingError) {
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
      await new Promise(resolve => setTimeout(resolve, 150));
    }

    chrome.tabs.sendMessage(tab.id, { action: "fetchActiveVideoDOM" }, (response) => {
      if (chrome.runtime.lastError) {
        activeVideoMetadata = { url: tab.url, title: tab.title || "YouTube Video", channel: "YouTube Creator" };
        updateUIElements(activeVideoMetadata.title, activeVideoMetadata.channel);
        disableActionControls(false);
        return;
      }

      if (response && response.title) {
        activeVideoMetadata = response;
        updateUIElements(response.title, response.channel);
        disableActionControls(false);
      }
    });

  } catch (err) {
    console.warn("Context layer recovery execution deferred:", err);
  }
}

function updateUIElements(title, channel) {
  document.getElementById("target-title").innerText = title;
  document.getElementById("target-channel").innerText = channel;
}

function renderOfflineState() {
  activeVideoMetadata = null;
  updateUIElements("Active system offline.", "Navigate to a valid YouTube watch link.");
  disableActionControls(true);
}

function disableActionControls(state) {
  document.getElementById("action-summarize").disabled = state;
  document.getElementById("action-save").disabled = state;
}

try {
  const port = chrome.runtime.connect({ name: "intelligence-sidepanel-pipe" });
  port.onMessage.addListener((message) => {
    if (message.type === "CONTEXT_UPDATE") {
      activeVideoMetadata = message.data;
      updateUIElements(message.data.title, message.data.channel);
      disableActionControls(false);
    } else if (message.type === "CONTEXT_OFFLINE") {
      forceSyncActiveTabContext();
    }
  });
} catch (e) {
  console.warn("Pipe link deferred.");
}

document.getElementById("action-summarize").addEventListener("click", async () => {
  if (!activeVideoMetadata) return;
  
  const outputArea = document.getElementById("output-area");
  outputArea.innerText = "Connecting to backend engine... Processing transcript segments via Gemini AI Pipeline...";

  try {
    const response = await fetch(`${BACKEND_ENDPOINT}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(activeVideoMetadata)
    });

    const parsedData = await response.json();
    if (parsedData.error) throw new Error(parsedData.error);

    // FIX: Swapped to innerText to prevent raw HTML rendering and map spacing text perfectly
    outputArea.innerText = parsedData.summary;
  } catch (err) {
    outputArea.innerText = `Pipeline Network Error: ${err.message}`;
  }
});

document.getElementById("action-save").addEventListener("click", async () => {
  if (!activeVideoMetadata) return;

  const notesText = document.getElementById("engineer-input").value;
  const aiSummaryText = document.getElementById("output-area").innerText;
  const statusDisplay = document.getElementById("status-display");

  statusDisplay.style.color = "#2563eb";
  statusDisplay.innerText = "Transmitting records...";

  try {
    const response = await fetch(`${BACKEND_ENDPOINT}/save-notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...activeVideoMetadata,
        summary: aiSummaryText,
        notes: notesText
      })
    });

    const resultData = await response.json();
    if (resultData.success) {
      statusDisplay.style.color = "#10b981";
      statusDisplay.innerText = "Saved securely via Firebase Access Layer.";
      document.getElementById("engineer-input").value = "";
    } else {
      throw new Error(resultData.error);
    }
  } catch (err) {
    statusDisplay.style.color = "#ef4444";
    statusDisplay.innerText = `Save Failure: ${err.message}`;
  }
  
  setTimeout(() => { statusDisplay.innerText = ""; }, 4000);
});

forceSyncActiveTabContext();
setInterval(forceSyncActiveTabContext, 2000);
