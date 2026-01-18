let session;
let fishClasses = [];

const predictionDiv = document.getElementById('prediction');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('preview-container');
const resultContainer = document.getElementById('result-container');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const imageUpload = document.getElementById('imageUpload');

async function init() {
    try {
        // Blocco iniziale
        predictBtn.disabled = true;
        resultContainer.classList.remove('hidden');
        predictionDiv.innerText = "⏳ Caricamento AI... attendere.";
        predictionDiv.style.color = "#e67e22";

        // Caricamento Classi
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n').map(s => s.trim()).filter(s => s !== "");

        // Caricamento Modello
        const [modelRes, dataRes] = await Promise.all([
            fetch('./model/model.onnx'),
            fetch('./model/model.onnx.data')
        ]);

        const modelBuffer = await modelRes.arrayBuffer();
        const dataBuffer = await dataRes.arrayBuffer();

        // Inizializzazione Sessione
        session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
            executionProviders: ['wasm'],
            externalData: [{ data: new Uint8Array(dataBuffer), path: "model.onnx.data" }]
        });

        // SBLOCCO SISTEMA
        predictBtn.disabled = false;
        predictionDiv.innerText = "✅ Sistema pronto.";
        predictionDiv.style.color = "#27ae60";

    } catch (e) {
        console.error(e);
        predictionDiv.innerText = "❌ Errore: " + e.message;
        predictionDiv.style.color = "#c0392b";
    }
}

predictBtn.addEventListener('click', async () => {
    if (!session) return;
    
    predictionDiv.innerText = "Analisi...";
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
        const nomeSpecie = fishClasses[maxIdx] || "Non identificato";

        predictionDiv.innerText = nomeSpecie;
        predictionDiv.style.opacity = "1";
        predictionDiv.style.color = "#2980b9";

    } catch (e) {
        predictionDiv.innerText = "Errore analisi.";
    }
});

resetBtn.addEventListener('click', () => {
    imageUpload.value = "";
    previewContainer.classList.add('hidden');
    predictionDiv.innerText = "✅ Sistema pronto.";
    predictionDiv.style.color = "#27ae60";
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
        reader.onload = (ev) => { 
            imagePreview.src = ev.target.result; 
            previewContainer.classList.remove('hidden'); 
        };
        reader.readAsDataURL(file);
    }
});

init();