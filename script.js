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

        const modelBuffer = await modelRes.arrayBuffer();
        const dataBuffer = await dataRes.arrayBuffer();

        console.log("3. Creazione sessione...");
        session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
            executionProviders: ['wasm'],
            externalData: [{ data: new Uint8Array(dataBuffer), path: "model.onnx.data" }]
        });

        // --- RIGA INVESTIGATIVA ---
        console.log("Nomi input attesi dal modello:", session.inputNames); 
        console.log("✅ SISTEMA PRONTO!");
        predictionDiv.innerText = "Sistema pronto.";
    } catch (e) {
        console.error("Errore:", e);
        predictionDiv.innerText = "Errore: " + e.message;
    }
}

document.getElementById('predictBtn').addEventListener('click', async () => {
    if (!session) return;
    predictionDiv.innerText = "Analisi...";
    resultContainer.classList.remove('hidden');

    try {
        const tensor = await preprocess(imagePreview);
        
        // Usiamo il primo nome di input che il modello ci dice di volere
        const inputName = session.inputNames[0]; 
        const feeds = {};
        feeds[inputName] = tensor;
        
        console.log("Invio dati all'input:", inputName);

        const results = await session.run(feeds);
        const outputKey = session.outputNames[0];
        const output = results[outputKey].data;
        
        const maxIdx = output.indexOf(Math.max(...output));
        predictionDiv.innerText = "Risultato: " + (fishClasses[maxIdx] || "ID: " + maxIdx);

    } catch (e) {
        console.error("Errore analisi:", e);
        predictionDiv.innerText = "Errore: " + e.message;
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
    return new ort.Tensor('float32', new Float32Array([...r, ...g, ...b]), [1, 3, 72, 256]);
}

document.getElementById('imageUpload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => { imagePreview.src = ev.target.result; document.getElementById('preview-container').classList.remove('hidden'); };
        reader.readAsDataURL(file);
    }
});

init();