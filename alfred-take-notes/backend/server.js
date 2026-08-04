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

const SERVICE_ACCOUNT_PATH = process.env.FIREBASE_SERVICE_ACCOUNT_PATH || './firebase-service-account.json';
let db = null;

try {
  if (fs.existsSync(SERVICE_ACCOUNT_PATH)) {
    const serviceAccount = JSON.parse(fs.readFileSync(SERVICE_ACCOUNT_PATH, 'utf8'));
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    db = admin.firestore();
    console.log("🔒 Firebase Admin operational container initialized successfully.");
  } else {
    console.warn("⚠️ Firebase service account file missing. Local mock state active.");
  }
} catch (firebaseInitError) {
  console.error("Firebase startup warning:", firebaseInitError.message);
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

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
      const videoId = parsedUrl.pathname.substring(1).split(/[?#]/);
      if (videoId && videoId.length === 11) return videoId;
    }
    return null;
  } catch (urlParsingError) {
    return null;
  }
}

app.post('/api/summarize', async (req, res) => {
  try {
    const { url, title, channel } = req.body;
    console.log(`📥 Processing execution request for: "${title}"`);

    if (!url) return res.status(400).json({ error: "Missing required video URL parameters." });

    const videoId = extractVideoId(url);
    if (!videoId) return res.status(400).json({ error: "Invalid YouTube URL format." });

    let transcriptText = "";
    let transcriptionSuccess = false;

    try {
      const transcriptPieces = await YoutubeTranscript.fetchTranscript(videoId);
      transcriptText = transcriptPieces.map(item => item.text).join(" ");
      transcriptionSuccess = true;
    } catch (transcriptError) {
      transcriptText = `Title context: ${title}. Channel: ${channel}.`;
    }

    const optimizedContext = transcriptText.substring(0, 50000);

    // CRITICAL FIX: The prompt forbids HTML/Markdown and mandates absolute spacing layouts
    const systemInstruction = transcriptionSuccess 
      ? "You are an expert enterprise research analyst. Analyze the provided transcript. " +
        "You must return your entire response as raw plain-text only. " +
        "Do not include any HTML elements, HTML tags, or markdown stars/hashes (like ### or **). " +
        "Organize the content strictly into the following three plain text blocks separated by empty lines:\n\n" +
        "CORE STRATEGY\n[Write a neat executive summary paragraph, followed by clean list items starting with simple dashes like '- Item']\n\n" +
        "TECHNICAL MILESTONES\n[Provide clear list items starting with simple dashes like '- Item']\n\n" +
        "ACTION ITEMS\n[Provide clear list items starting with simple dashes like '- Item']"
      : "You are an expert enterprise assistant. Transcripts are disabled. " +
        "Analyze the video title and channel. Provide an intelligent overview using clean headers and plain dash bullets.";

    const aiResponse = await ai.models.generateContent({
      model: 'gemini-3.6-flash',
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
      console.log("💾 Mock Save Action Active:\n", { url, title, notes });
      return res.status(200).json({ success: true, docId: "mock-simulated-id" });
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
    return res.status(500).json({ error: "Failed to persist document container." });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Free Gemini AI Summary Engine active on port ${PORT}`));
