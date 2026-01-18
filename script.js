let session;
let fishClasses = [];

const predictionDiv = document.getElementById('prediction');
const imagePreview = document.getElementById('imagePreview');
const resultContainer = document.getElementById('result-container');

async function init() {
    try {
        console.log("Inizio caricamento avanzato...");

        // 1. Carica Classi
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        // 2. Carica i due file del modello come BLOB
        console.log("Scaricamento file binari...");
        const [modelRes, dataRes] = await Promise.all([
            fetch('./model/model.onnx'),
            fetch('./model/model.onnx.data')
        ]);

        if (!modelRes.ok || !dataRes.ok) throw new Error("File del modello non trovati nella cartella /model");

        const modelBlob = await modelRes.blob();
        const dataBlob = await dataRes.blob();

        // 3. Creazione filesystem virtuale (Trick per MountedFiles)
        // Creiamo dei file virtuali che ONNX Runtime può vedere
        const modelUrl = URL.createObjectURL(modelBlob);
        const dataUrl = URL.createObjectURL(dataBlob);

        console.log("Inizializzazione sessione WASM...");
        
        // Usiamo l'opzione manuale per i dati esterni
        session = await ort.InferenceSession.create(await modelBlob.arrayBuffer(), {
            executionProviders: ['wasm'],
            externalData: [
                {
                    data: await dataBlob.arrayBuffer(),
                    path: "model.onnx.data" // Deve essere ESATTAMENTE il nome cercato dal modello
                }
            ]
        });

        console.log("✅ SISTEMA PRONTO!");
        predictionDiv.innerText = "Sistema pronto per l'analisi.";
        
        // Pulizia memoria
        URL.revokeObjectURL(modelUrl);
        URL.revokeObjectURL(dataUrl);

    } catch (e) {
        console.error("Errore irreversibile:", e);
        predictionDiv.innerText = "Errore: " + e.message;
    }
}

// Logica di predizione (Resta invariata ma inclusa per completezza)
document.getElementById('predictBtn').addEventListener('click', async () => {
    if (!session) return;
    predictionDiv.innerText = "Analisi...";
    resultContainer.classList.remove('hidden');

    try {
        const tensor = await preprocess(imagePreview);
        const results = await session.run({ input: tensor });
        const output = results.output.data;
        const maxIdx = output.indexOf(Math.max(...output));
        predictionDiv.innerText = fishClasses[maxIdx] || "Sconosciuto";
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

document.getElementById('imageUpload').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => { imagePreview.src = ev.target.result; document.getElementById('preview-container').classList.remove('hidden'); };
        reader.readAsDataURL(file);
    }
});

init();