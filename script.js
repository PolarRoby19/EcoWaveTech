let session;
let fishClasses = [];

// Elementi UI
const predictionDiv = document.getElementById('prediction');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('preview-container');
const resultContainer = document.getElementById('result-container');
const predictBtn = document.getElementById('predictBtn');
const imageUpload = document.getElementById('imageUpload');

/**
 * Inizializzazione con indicatore di caricamento
 */
async function init() {
    try {
        predictBtn.disabled = true; // Disabilita il tasto finché non è pronto
        predictionDiv.innerText = "⏳ Caricamento modelli in corso... attendere.";
        predictionDiv.style.color = "#ff9800"; // Arancione durante l'attesa

        console.log("1. Caricamento classi...");
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        console.log("2. Scaricamento pesi e struttura...");
        const [modelRes, dataRes] = await Promise.all([
            fetch('./model/model.onnx'),
            fetch('./model/model.onnx.data')
        ]);

        const modelBuffer = await modelRes.arrayBuffer();
        const dataBuffer = await dataRes.arrayBuffer();

        console.log("3. Creazione sessione...");
        session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
            executionProviders: ['wasm'],
            externalData: [{ data: new Uint8Array(dataBuffer), path: "model.onnx.data" }]
        });

        console.log("✅ SISTEMA PRONTO!");
        predictionDiv.innerText = "✅ Sistema pronto. Carica un'immagine.";
        predictionDiv.style.color = "#4caf50"; // Verde quando è pronto
        predictBtn.disabled = false;

    } catch (e) {
        console.error("Errore inizializzazione:", e);
        predictionDiv.innerText = "❌ Errore caricamento: " + e.message;
        predictionDiv.style.color = "#f44336";
    }
}

/**
 * Logica di analisi
 */
predictBtn.addEventListener('click', async () => {
    if (!session) return;
    
    predictionDiv.style.opacity = "0.5";
    predictionDiv.innerText = "Analisi in corso...";
    resultContainer.classList.remove('hidden');

    try {
        const tensor = await preprocess(imagePreview);
        const inputName = session.inputNames[0]; 
        const feeds = {};
        feeds[inputName] = tensor;
        
        const results = await session.run(feeds);
        const outputKey = session.outputNames[0];
        const output = results[outputKey].data;
        
        const maxIdx = output.indexOf(Math.max(...output));
        const nomeSpecie = fishClasses[maxIdx] || "ID: " + maxIdx;

        predictionDiv.innerText = nomeSpecie;
        predictionDiv.style.color = "#007bff";
        predictionDiv.style.transition = "opacity 0.5s";
        predictionDiv.style.opacity = "1";

    } catch (e) {
        console.error("Errore analisi:", e);
        predictionDiv.innerText = "Errore: " + e.message;
        predictionDiv.style.opacity = "1";
    }
});

/**
 * Funzione Reset (da aggiungere un tasto con id="resetBtn" nell'HTML se vuoi usarlo)
 */
function resetAll() {
    imageUpload.value = "";
    previewContainer.classList.add('hidden');
    resultContainer.classList.add('hidden');
    predictionDiv.innerText = "✅ Sistema pronto. Carica un'immagine.";
    predictionDiv.style.color = "#4caf50";
}

// Se aggiungi <button id="resetBtn"> nell'HTML:
const resetBtn = document.getElementById('resetBtn');
if (resetBtn) {
    resetBtn.addEventListener('click', resetAll);
}

/**
 * Pre-processing
 */
async function preprocess(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256; canvas.height = 72;
    ctx.drawImage(img, 0, 0, 256, 72);
    const data = ctx.getImageData(0, 0, 256, 72).data;
    const r = [], g = [], b = [];
    for (let i = 0; i < data.length; i += 4) {
        r.push(data[i] / 255.0);
        g.push(data[i+1] / 255.0);
        b.push(data[i+2] / 255.0);
    }
    return new ort.Tensor('float32', new Float32Array([...r, ...g, ...b]), [1, 3, 72, 256]);
}

/**
 * Upload immagine
 */
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => { 
            imagePreview.src = ev.target.result; 
            previewContainer.classList.remove('hidden'); 
            resultContainer.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
});

init();