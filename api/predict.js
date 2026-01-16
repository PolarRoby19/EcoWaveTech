export const config = {
  api: {
    bodyParser: false, // Necessario per leggere l'immagine come stream grezzo
  },
};

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Metodo non consentito' });
    }

    const HF_TOKEN = process.env.HF_TOKEN;
    const MODEL_ID = "PolarRoby09/ecowave";

    // Controllo sicurezza: il token esiste?
    if (!HF_TOKEN) {
        return res.status(500).json({ error: "Configurazione mancante: HF_TOKEN non trovato su Vercel" });
    }

    try {
        // Leggiamo i dati dell'immagine
        const chunks = [];
        for await (const chunk of req) {
            chunks.push(chunk);
        }
        const buffer = Buffer.concat(chunks);

        // Chiamata a Hugging Face
        const response = await fetch(
            `https://api-inference.huggingface.co/models/${MODEL_ID}`,
            {
                headers: {
                    "Authorization": `Bearer ${HF_TOKEN}`,
                    "Content-Type": "application/octet-stream",
                },
                method: "POST",
                body: buffer,
            }
        );

        const data = await response.json();

        if (!response.ok) {
            // Passiamo l'errore originale di Hugging Face per capire cosa succede
            return res.status(response.status).json(data);
        }

        return res.status(200).json(data);
    } catch (error) {
        console.error("ERRORE SERVERLESS:", error);
        return res.status(500).json({ error: "Errore durante l'invio all'IA: " + error.message });
    }
}