let session;
let fishClasses = [];

// Elementi UI
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('preview-container');
const predictBtn = document.getElementById('predictBtn');
const predictionDiv = document.getElementById('prediction');
const resultContainer = document.getElementById('result-container');

// 1. Carica Modello e Nomi Classi
async function init() {
    try {
        // Carica i nomi dal CSV
        const response = await fetch('classes.csv');
        const data = await response.text();
        fishClasses = data.split('\n').map(line => line.trim()).filter(line => line !== "");
        
        // Carica il modello ONNX
        // Nota: ONNX cercherà automaticamente il file .data nella stessa cartella
        session = await ort.InferenceSession.create('./model/model.onnx');
        
        console.log("Sistema pronto. Classi caricate:", fishClasses.length);
    } catch (e) {
        console.error("Errore inizializzazione:", e);
    }
}

// 2. Gestione Anteprima
imageUpload.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            resultContainer.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
});

// 3. Analisi Immagine
predictBtn.addEventListener('click', async () => {
    predictionDiv.innerText = "Analisi in corso...";
    resultContainer.classList.remove('hidden');
    
    try {
        const inputTensor = await preprocess(imagePreview);
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        
        // L'output del tuo modello si chiama solitamente 'output'
        // Se dà errore, prova a stampare 'console.log(results)' per vedere il nome esatto
        const output = results.output.data; 
        const maxIndex = argmax(output);
        
        const fishName = fishClasses[maxIndex] || "Specie sconosciuta";
        predictionDiv.innerText = fishName;
        
    } catch (e) {
        predictionDiv.innerText = "Errore durante l'analisi.";
        console.error(e);
    }
});

// 4. Pre-processing (Resize 256x72 + Normalizzazione)
async function preprocess(imgElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 72;
    ctx.drawImage(imgElement, 0, 0, 256, 72);
    
    const imageData = ctx.getImageData(0, 0, 256, 72).data;
    const r = [], g = [], b = [];

    for (let i = 0; i < imageData.length; i += 4) {
        r.push(imageData[i] / 255.0);
        g.push(imageData[i + 1] / 255.0);
        b.push(imageData[i + 2] / 255.0);
    }

    return new ort.Tensor('float32', new Float32Array([...r, ...g, ...b]), [1, 3, 72, 256]);
}

function argmax(array) {
    return array.indexOf(Math.max(...array));
}

init();