export const config = {
  api: {
    bodyParser: false, // Disabilitiamo il parser automatico per gestire i dati grezzi dell'immagine
  },
};

export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Metodo non consentito' });
    }

    const HF_TOKEN = process.env.HF_TOKEN;
    const MODEL_ID = "PolarRoby09/ecowave";

    try {
        // Leggiamo i dati dell'immagine come chunk e creiamo un Buffer
        const chunks = [];
        for await (const chunk of req) {
            chunks.push(chunk);
        }
        const buffer = Buffer.concat(chunks);

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
            return res.status(response.status).json(data);
        }

        return res.status(200).json(data);
    } catch (error) {
        console.error("Errore:", error);
        return res.status(500).json({ error: "Errore interno durante l'analisi" });
    }
}