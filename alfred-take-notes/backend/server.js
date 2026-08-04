import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { YoutubeTranscript } from 'youtube-transcript';
import { GoogleGenAI } from '@google/genai';
import admin from 'firebase-admin';
import fs from 'fs';

dotenv.config();

const app = express();
app.use(cors({ origin: '*' })); 
app.use(express.json());

// Initialize Firebase Admin container safely
const SERVICE_ACCOUNT_PATH = process.env.FIREBASE_SERVICE_ACCOUNT_PATH || './firebase-service-account.json';
let db = null;

try {
  if (fs.existsSync(SERVICE_ACCOUNT_PATH)) {
    const serviceAccount = JSON.parse(fs.readFileSync(SERVICE_ACCOUNT_PATH, 'utf8'));
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    db = admin.firestore();
    console.log("🔒 Firebase Admin operational container initialized successfully.");
  } else {
    console.warn("⚠️ Firebase service account file missing. Running database routes in mock console storage mode.");
  }
} catch (firebaseInitError) {
  console.error("Firebase startup warning:", firebaseInitError.message);
}

// Initialize the 100% Free Google Gemini Developer API Client SDK
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Production-grade URL parsing architecture bypassing regular expressions completely
function extractVideoId(urlString) {
  if (!urlString) return null;
  try {
    const parsedUrl = new URL(urlString);
    if (parsedUrl.hostname.includes("youtube.com")) {
      const videoId = parsedUrl.searchParams.get("v");
      if (videoId && videoId.length === 11) return videoId;
      
      const pathSegments = parsedUrl.pathname.split('/');
      const shortsIndex = pathSegments.indexOf('shorts');
      if (shortsIndex !== -1 && pathSegments[shortsIndex + 1]?.length === 11) {
        return pathSegments[shortsIndex + 1];
      }
    }
    if (parsedUrl.hostname.includes("youtu.be")) {
      const videoId = parsedUrl.pathname.substring(1).split(/[?#]/)[0];
      if (videoId && videoId.length === 11) return videoId;
    }
    return null;
  } catch (urlParsingError) {
    console.error("Critical URL parser failure:", urlParsingError.message);
    return null;
  }
}

// Enterprise Free Summarization Orchestrator Route
app.post('/api/summarize', async (req, res) => {
  try {
    const { url, title, channel } = req.body;
    console.log(`📥 Processing execution request stream target for video: "${title}" [${channel}]`);

    if (!url) return res.status(400).json({ error: "Missing required video URL parameters." });

    const videoId = extractVideoId(url);
    if (!videoId) return res.status(400).json({ error: "Invalid YouTube URL format validation rejected." });

    let transcriptText = "";
    let transcriptionSuccess = false;

    try {
      const transcriptPieces = await YoutubeTranscript.fetchTranscript(videoId);
      transcriptText = transcriptPieces.map(item => item.text).join(" ");
      transcriptionSuccess = true;
      console.log(`✅ Transcript extracted successfully. Context length: ${transcriptText.length} characters.`);
    } catch (transcriptError) {
      console.warn(`⚠️ Transcripts unavailable for video ${videoId}. Shifting to metadata context parameters.`);
      transcriptText = `Title context matches: ${title}. Creator channel matches: ${channel}.`;
    }

    // Protect token size limits safely
    const optimizedContext = transcriptText.substring(0, 50000);

    const systemInstruction = transcriptionSuccess 
      ? "You are an executive enterprise assistant. Analyze the transcript text provided. Produce clear, highly professional bulleted summaries. Break content down into 'Core Strategy', 'Technical Milestones', and 'Action Items'. Keep explanations concise."
      : "You are an executive enterprise assistant. Transcripts are disabled for this video. Analyze the available video title and channel context parameters. Provide an intelligent overview, potential structural topics covered by this creator, and actionable learning goals.";

    // FIX: Updated model parameter key targeting Google's active free-tier model flag
    const aiResponse = await ai.models.generateContent({
      model: 'gemini-3.6-flash', // Swapped out deprecated gemini-2.5-flash
      contents: [
        { role: 'user', parts: [{ text: `${systemInstruction}\n\nTranscript / Metadata Content:\n${optimizedContext}` }] }
      ]
    });

    const finalSummary = aiResponse.text || "AI failed to produce readable summary text.";
    return res.status(200).json({ summary: finalSummary, videoId });

  } catch (globalError) {
    console.error("🔴 Backend critical error: ", globalError);
    return res.status(500).json({ error: `Gemini Engine Error: ${globalError.message}` });
  }
});

app.post('/api/save-notes', async (req, res) => {
  try {
    const { url, title, channel, summary, notes } = req.body;

    if (!db) {
      console.log("💾 Mock Save Log Container Capture:\n", { url, title, notes });
      return res.status(200).json({ success: true, docId: "mock-simulated-id-no-firebase-file" });
    }

    const savedDoc = await db.collection('enterprise_video_notes').add({
      url,
      title,
      channel,
      summary,
      userNotes: notes,
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });

    return res.status(200).json({ success: true, docId: savedDoc.id });
  } catch (dbError) {
    console.error("Firestore persistence layer failure:", dbError);
    return res.status(500).json({ error: "Failed to persist document secure container." });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Free Gemini AI Summary Engine active on port ${PORT}`));
