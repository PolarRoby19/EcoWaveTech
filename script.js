let session;
let fishClasses = [];

// Elementi UI
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('preview-container');
const predictBtn = document.getElementById('predictBtn');
const predictionDiv = document.getElementById('prediction');
const resultContainer = document.getElementById('result-container');

/**
 * Inizializzazione: Carica classi e modello.
 * Usa l'approccio arrayBuffer per evitare errori di caricamento dati esterni.
 */
async function init() {
    try {
        console.log("1. Caricamento classi...");
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        console.log("2. Scaricamento pesi (model.onnx.data)...");
        // Scarichiamo prima i pesi per assicurarci che siano in cache
        const dataRes = await fetch('./model/model.onnx.data');
        if (!dataRes.ok) throw new Error("Impossibile trovare model.onnx.data");
        const dataBuffer = await dataRes.arrayBuffer();

        console.log("3. Scaricamento struttura (model.onnx)...");
        const modelRes = await fetch('./model/model.onnx');
        if (!modelRes.ok) throw new Error("Impossibile trovare model.onnx");
        const modelBuffer = await modelRes.arrayBuffer();

        console.log("4. Creazione sessione...");
        // Configuriamo ONNX per usare i dati esterni forniti manualmente
        session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            externalData: [
                {
                    data: dataBuffer,
                    path: "model.onnx.data" // Deve corrispondere al nome cercato dal modello
                }
            ]
        });

        console.log("✅ Sistema pronto!");
        predictionDiv.innerText = "Sistema pronto.";
    } catch (e) {
        console.error("Errore critico:", e);
        predictionDiv.innerText = "Errore: " + e.message;
    }
}

/**
 * Gestione Anteprima Immagine
 */
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            previewContainer.classList.remove('hidden');
            resultContainer.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
});

/**
 * Logica di Predizione
 */
predictBtn.addEventListener('click', async () => {
    if (!session) {
        alert("Il modello si sta ancora caricando. Attendi...");
        return;
    }

    predictionDiv.innerText = "Analisi in corso...";
    resultContainer.classList.remove('hidden');

    try {
        // Pre-processing immagine
        const tensor = await preprocess(imagePreview);
        
        // Esecuzione Inferenza
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        
        // Estrazione risultati (usa 'output' come nome predefinito di PyTorch)
        const output = results.output.data;
        const maxIdx = argmax(output);
        
        // Mostra il nome della specie dal CSV
        const fishName = fishClasses[maxIdx] || "ID: " + maxIdx + " (Nome non in CSV)";
        predictionDiv.innerText = fishName;
        
    } catch (e) {
        console.error("Errore predizione:", e);
        predictionDiv.innerText = "Errore durante l'analisi.";
    }
});

/**
 * Pre-processing: Ridimensiona a 256x72 e normalizza i canali RGB
 */
async function preprocess(imgElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const w = 256;
    const h = 72;
    
    canvas.width = w;
    canvas.height = h;
    ctx.drawImage(imgElement, 0, 0, w, h);
    
    const imageData = ctx.getImageData(0, 0, w, h).data;
    const r = new Float32Array(w * h);
    const g = new Float32Array(w * h);
    const b = new Float32Array(w * h);

    for (let i = 0, j = 0; i < imageData.length; i += 4, j++) {
        r[j] = imageData[i] / 255.0;
        g[j] = imageData[i + 1] / 255.0;
        b[j] = imageData[i + 2] / 255.0;
    }

    const float32Data = new Float32Array([...r, ...g, ...b]);
    return new ort.Tensor('float32', float32Data, [1, 3, h, w]);
}

function argmax(array) {
    return array.indexOf(Math.max(...array));
}

// Avvio
init();