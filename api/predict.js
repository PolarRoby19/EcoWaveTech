export const config = {
  api: { bodyParser: false },
};

export default async function handler(req, res) {
    const HF_TOKEN = process.env.HF_TOKEN;
    const MODEL_ID = "PolarRoby09/EcoWave-Vercel";

    try {
        const chunks = [];
        for await (const chunk of req) { chunks.push(chunk); }
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

        const contentType = response.headers.get("content-type");
        
        // Se la risposta non è JSON, c'è un errore di configurazione
        if (!contentType || !contentType.includes("application/json")) {
            const errorText = await response.text();
            console.error("Risposta non JSON ricevuta:", errorText);
            return res.status(500).json({ error: "L'IA ha risposto con una pagina HTML. Verifica il Token e l'URL del modello." });
        }

        const data = await response.json();
        return res.status(response.status).json(data);

    } catch (error) {
        return res.status(500).json({ error: "Errore fatale: " + error.message });
    }
}