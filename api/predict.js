export default async function handler(req, res) {
    // 1. Accetta solo richieste POST (invio immagini)
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Metodo non consentito' });
    }

    // 2. Recupera le variabili che hai impostato su Vercel
    const HF_TOKEN = process.env.HF_TOKEN; 
    // Sostituisci con il tuo username e nome repo di Hugging Face
    const MODEL_ID = "IL_TUO_USERNAME_HF/IL_TUO_REPO_HF"; 

    try {
        // 3. Inoltra l'immagine a Hugging Face usando il Token segreto
        const response = await fetch(
            `https://api-inference.huggingface.co/models/${MODEL_ID}`,
            {
                headers: { 
                    "Authorization": `Bearer ${HF_TOKEN}`,
                    "Content-Type": "application/octet-stream"
                },
                method: "POST",
                body: req.body, // L'immagine grezza ricevuta dal sito
            }
        );

        const data = await response.json();
        
        // 4. Restituisce il risultato al tuo sito
        return res.status(200).json(data);
    } catch (error) {
        return res.status(500).json({ error: "Errore nella comunicazione con l'IA" });
    }
}