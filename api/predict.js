export const config = {
  api: { bodyParser: false },
};

export default async function handler(req, res) {
    const HF_TOKEN = process.env.HF_TOKEN;
    const MODEL_ID = "PolarRoby09/EcoWave-Vercel";

console.log("DEBUG - Lunghezza Token:", HF_TOKEN ? HF_TOKEN.length : "VUOTO!");}