let session;
let fishClasses = [];

const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('preview-container');
const predictBtn = document.getElementById('predictBtn');
const predictionDiv = document.getElementById('prediction');
const resultContainer = document.getElementById('result-container');

async function init() {
    try {
        console.log("1. Caricamento classi...");
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        console.log("2. Scaricamento pesi (model.onnx.data)...");
        const dataRes = await fetch('./model/model.onnx.data');
        if (!dataRes.ok) throw new Error("File .data non trovato");
        const dataBuffer = await dataRes.arrayBuffer();

        console.log("3. Scaricamento struttura (model.onnx)...");
        const modelRes = await fetch('./model/model.onnx');
        if (!modelRes.ok) throw new Error("File .onnx non trovato");
        const modelBuffer = await modelRes.arrayBuffer();

        console.log("4. Creazione sessione con iniezione dati...");
        
        // Questa configurazione bypassa l'errore Module.MountedFiles
        session = await ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            externalData: [
                {
                    data: dataBuffer,
                    path: "model.onnx.data" // Deve corrispondere esattamente al nome cercato dal modello
                }
            ]
        });

        console.log("✅ Sistema pronto!");
        predictionDiv.innerText = "Sistema pronto.";
    } catch (e) {
        console.error("Errore critico:", e);
        predictionDiv.innerText = "Errore caricamento: " + e.message;
    }
}

// Pre-processing e Predizione
predictBtn.addEventListener('click', async () => {
    if (!session) return;
    predictionDiv.innerText = "Analisi...";
    resultContainer.classList.remove('hidden');

    try {
        const tensor = await preprocess(imagePreview);
        const results = await session.run({ input: tensor });
        const output = results.output.data;
        const maxIdx = output.indexOf(Math.max(...output));
        
        predictionDiv.innerText = fishClasses[maxIdx] || "ID: " + maxIdx;
    } catch (e) {
        console.error(e);
        predictionDiv.innerText = "Errore analisi.";
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

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => { imagePreview.src = ev.target.result; previewContainer.classList.remove('hidden'); };
        reader.readAsDataURL(file);
    }
});

init();