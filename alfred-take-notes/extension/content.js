function grabYoutubeMetadata() {
  const title = document.querySelector("h1 yt-formatted-string")?.textContent?.trim() ||
                document.querySelector("h1.ytd-watch-metadata yt-formatted-string")?.textContent?.trim() ||
                document.title.replace(" - YouTube", "").trim();

  const channel = document.querySelector("#channel-name a")?.textContent?.trim() ||
                  document.querySelector("#owner #text")?.textContent?.trim() ||
                  document.querySelector("#channel-name #text")?.textContent?.trim() ||
                  "YouTube Creator";

  return {
    // Explicitly targets the clean global URL from the master frame context
    url: window.top.location.href.split('&')[0], // Automatically strips out tracking attributes
    title,
    channel
  };
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "ping") {
    sendResponse({ alive: true });
  } else if (request.action === "fetchActiveVideoDOM") {
    sendResponse(grabYoutubeMetadata());
  }
  return true; 
});

let lastUrl = window.location.href;
const observer = new MutationObserver(() => {
  if (window.location.href !== lastUrl && window.location.href.includes("youtube.com/watch")) {
    lastUrl = window.location.href;
    
    setTimeout(() => {
      try {
        chrome.runtime.sendMessage({ 
          type: "DYNAMIC_VIDEO_CHANGE", 
          data: grabYoutubeMetadata() 
        });
      } catch (e) {
        // Safe defer if channel isn't bound yet
      }
    }, 1200);
  }
});

function initializeObserver() {
  if (document.body) {
    observer.observe(document.body, { childList: true, subtree: true });
  } else {
    document.addEventListener("DOMContentLoaded", () => {
      if (document.body) observer.observe(document.body, { childList: true, subtree: true });
    });
  }
}

initializeObserver();
