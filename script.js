let session;
let fishClasses = [];

async function init() {
    const predictionDiv = document.getElementById('prediction');
    try {
        console.log("1. Caricamento nomi classi...");
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        console.log("2. Scaricamento pesi (model.onnx.data)...");
        const dataRes = await fetch('./model/model.onnx.data');
        if (!dataRes.ok) throw new Error("File model.onnx.data non trovato");
        const dataBuffer = await dataRes.arrayBuffer();

        console.log("3. Scaricamento struttura (model.onnx)...");
        const modelRes = await fetch('./model/model.onnx');
        if (!modelRes.ok) throw new Error("File model.onnx non trovato");
        const modelBuffer = await modelRes.arrayBuffer();

        console.log("4. Iniezione dati e creazione sessione...");
        
        // Questo oggetto mappa il nome del file interno al buffer scaricato
        const externalData = [
            {
                data: new Uint8Array(dataBuffer),
                path: 'model.onnx.data' // Deve essere identico al nome cercato dal modello
            }
        ];

        session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
            executionProviders: ['wasm'],
            externalData: externalData
        });

        console.log("✅ SISTEMA PRONTO!");
        predictionDiv.innerText = "Sistema pronto.";
    } catch (e) {
        console.error("Errore critico:", e);
        predictionDiv.innerText = "Errore: " + e.message;
    }
}

// --- Logica di analisi ---
document.getElementById('predictBtn').addEventListener('click', async () => {
    if (!session) return;
    const predictionDiv = document.getElementById('prediction');
    predictionDiv.innerText = "Analisi...";

    try {
        const tensor = await preprocess(document.getElementById('imagePreview'));
        const results = await session.run({ input: tensor });
        const output = results.output.data;
        const maxIdx = output.indexOf(Math.max(...output));
        predictionDiv.innerText = fishClasses[maxIdx] || "ID: " + maxIdx;
    } catch (e) {
        console.error(e);
        predictionDiv.innerText = "Errore durante l'analisi.";
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
        reader.onload = (ev) => {
            const img = document.getElementById('imagePreview');
            img.src = ev.target.result;
            document.getElementById('preview-container').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
});

init();