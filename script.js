let session;
let fishClasses = [];

// Elementi UI
const predictionDiv = document.getElementById('prediction');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('preview-container');
const resultContainer = document.getElementById('result-container');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const imageUpload = document.getElementById('imageUpload');

/**
 * Inizializzazione: Carica modelli e classi
 */
async function init() {
    try {
        // Mostriamo il caricamento fin dall'inizio
        resultContainer.classList.remove('hidden');
        predictionDiv.innerText = "⏳ Inizializzazione AI... attendere.";
        predictionDiv.style.color = "#ff9800";
        predictBtn.disabled = true;

        console.log("1. Caricamento nomi classi...");
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        console.log("2. Scaricamento file modello...");
        const [modelRes, dataRes] = await Promise.all([
            fetch('./model/model.onnx'),
            fetch('./model/model.onnx.data')
        ]);

        const modelBuffer = await modelRes.arrayBuffer();
        const dataBuffer = await dataRes.arrayBuffer();

        console.log("3. Creazione sessione ONNX...");
        session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
            executionProviders: ['wasm'],
            externalData: [{ data: new Uint8Array(dataBuffer), path: "model.onnx.data" }]
        });

        console.log("✅ SISTEMA PRONTO!");
        predictionDiv.innerText = "✅ Sistema pronto. Carica un pesce!";
        predictionDiv.style.color = "#4caf50";
        predictBtn.disabled = false;

    } catch (e) {
        console.error("Errore critico:", e);
        predictionDiv.innerText = "❌ Errore caricamento: " + e.message;
        predictionDiv.style.color = "#f44336";
    }
}

/**
 * Gestione Analisi
 */
predictBtn.addEventListener('click', async () => {
    if (!session) return;
    
    predictionDiv.innerText = "Analisi in corso...";
    predictionDiv.style.color = "#007bff";
    predictionDiv.style.opacity = "0.5";

    try {
        const tensor = await preprocess(imagePreview);
        const inputName = session.inputNames[0]; 
        const feeds = {};
        feeds[inputName] = tensor;
        
        const results = await session.run(feeds);
        const outputKey = session.outputNames[0];
        const output = results[outputKey].data;
        
        const maxIdx = output.indexOf(Math.max(...output));
        const nomeSpecie = fishClasses[maxIdx] || "Specie non riconosciuta";

        predictionDiv.innerText = nomeSpecie;
        predictionDiv.style.opacity = "1";
        predictionDiv.style.transition = "opacity 0.5s";

    } catch (e) {
        console.error("Errore:", e);
        predictionDiv.innerText = "Errore analisi.";
    }
});

/**
 * Funzione Reset
 */
resetBtn.addEventListener('click', () => {
    imageUpload.value = "";
    previewContainer.classList.add('hidden');
    resultContainer.classList.remove('hidden'); // Teniamo il messaggio di sistema pronto
    predictionDiv.innerText = "✅ Sistema pronto. Carica un'immagine.";
    predictionDiv.style.color = "#4caf50";
});

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
 * Caricamento Immagine e Visibilità Bottoni
 */
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => { 
            imagePreview.src = ev.target.result; 
            // Mostra il contenitore con anteprima e bottoni
            previewContainer.classList.remove('hidden'); 
            // Nasconde il risultato precedente mentre si cambia foto
            predictionDiv.innerText = "Pronto per l'analisi.";
            predictionDiv.style.color = "#4caf50";
        };
        reader.readAsDataURL(file);
    }
});

init();