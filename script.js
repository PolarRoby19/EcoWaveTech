let session;
let fishClasses = [];

const predictionDiv = document.getElementById('prediction');
const imagePreview = document.getElementById('imagePreview');
const resultContainer = document.getElementById('result-container');

async function init() {
    try {
        console.log("1. Caricamento classi...");
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        console.log("2. Scaricamento file binari...");
        const [modelRes, dataRes] = await Promise.all([
            fetch('./model/model.onnx'),
            fetch('./model/model.onnx.data')
        ]);

        if (!modelRes.ok || !dataRes.ok) throw new Error("File non trovati in /model");

        const modelBuffer = await modelRes.arrayBuffer();
        const dataBuffer = await dataRes.arrayBuffer();

        console.log("3. Iniezione dati e creazione sessione...");
        session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
            executionProviders: ['wasm'],
            externalData: [{ data: new Uint8Array(dataBuffer), path: "model.onnx.data" }]
        });

        console.log("✅ SISTEMA PRONTO!");
        predictionDiv.innerText = "Sistema pronto.";
    } catch (e) {
        console.error("Errore critico:", e);
        predictionDiv.innerText = "Errore: " + e.message;
    }
}

// --- Logica di analisi (MODIFICATA) ---
document.getElementById('predictBtn').addEventListener('click', async () => {
    if (!session) return;
    predictionDiv.innerText = "Analisi in corso...";
    resultContainer.classList.remove('hidden');

    try {
        const tensor = await preprocess(imagePreview);
        
        // QUI C'È LA MODIFICA: abbiamo cambiato 'input' in 'x'
        const feeds = { x: tensor }; 
        
        const results = await session.run(feeds);
        
        // Cerchiamo l'output (solitamente si chiama 'output' o 'output_0')
        const outputKey = Object.keys(results)[0]; 
        const output = results[outputKey].data;
        
        const maxIdx = output.indexOf(Math.max(...output));
        const fishName = fishClasses[maxIdx] || "ID: " + maxIdx;
        
        predictionDiv.innerText = "Risultato: " + fishName;
        console.log("Predizione completata:", fishName);

    } catch (e) {
        console.error("Errore durante l'analisi:", e);
        predictionDiv.innerText = "Errore durante l'analisi: " + e.message;
    }
});

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
    const float32Data = new Float32Array([...r, ...g, ...b]);
    return new ort.Tensor('float32', float32Data, [1, 3, 72, 256]);
}

document.getElementById('imageUpload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => { 
            imagePreview.src = ev.target.result; 
            document.getElementById('preview-container').classList.remove('hidden'); 
            resultContainer.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
});

init();