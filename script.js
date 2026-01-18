let session;
let fishClasses = [];

// Elementi UI
const predictionDiv = document.getElementById('prediction');

async function init() {
    try {
        console.log("Inizio caricamento...");

        // 1. Carica Classi
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        // 2. Carica Modello
        // Usiamo il percorso diretto senza './' per evitare problemi di root su GitHub Pages
        session = await ort.InferenceSession.create('model.onnx', {
            executionProviders: ['wasm']
        });

        console.log("✅ SESSIONE CREATA!");
        predictionDiv.innerText = "Sistema pronto.";
    } catch (e) {
        console.error("Errore critico:", e);
        // Se vedi ancora 'MountedFiles', il problema è l'esportazione del modello
        predictionDiv.innerText = "Errore: " + e.message;
    }
}

// ... (tieni il resto delle funzioni preprocess e i listener come prima)
init();