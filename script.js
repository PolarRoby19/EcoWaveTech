let session;
let fishClasses = [];

// Elementi dell'interfaccia utente
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('preview-container');
const predictBtn = document.getElementById('predictBtn');
const predictionDiv = document.getElementById('prediction');
const resultContainer = document.getElementById('result-container');

/**
 * 1. Inizializzazione: Carica i nomi delle classi e il modello ONNX.
 */
async function init() {
    try {
        console.log("Inizializzazione sistema...");

        // Carica i nomi dal file CSV (nella stessa cartella di script.js)
        const csvRes = await fetch('classes.csv');
        const csvText = await csvRes.text();
        fishClasses = csvText.split('\n')
            .map(s => s.trim())
            .filter(s => s !== "");
        
        console.log("Classi caricate:", fishClasses.length);

        // Carica il modello ONNX dalla cartella 'model'
        // NOTA: Il browser scaricherà automaticamente anche model.onnx.data 
        // perché è collegato internamente al file .onnx
        session = await ort.InferenceSession.create('./model/model.onnx');
        
        console.log("Modello ONNX caricato con successo!");
    } catch (e) {
        console.error("Errore durante l'inizializzazione:", e);
        predictionDiv.innerText = "Errore nel caricamento del sistema.";
    }
}

/**
 * 2. Gestione dell'anteprima dell'immagine caricata
 */
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            previewContainer.classList.remove('hidden');
            resultContainer.classList.add('hidden'); // Nasconde risultati precedenti
        };
        reader.readAsDataURL(file);
    }
});

/**
 * 3. Esecuzione della predizione al click sul pulsante
 */
predictBtn.addEventListener('click', async () => {
    if (!session) {
        alert("Il modello non è ancora pronto. Attendi qualche secondo.");
        return;
    }

    predictionDiv.innerText = "Analisi in corso...";
    resultContainer.classList.remove('hidden');

    try {
        // Pre-processing dell'immagine per adattarla al modello (1, 3, 72, 256)
        const tensor = await preprocess(imagePreview);
        
        // Esecuzione dell'inferenza
        const feeds = { input: tensor };
        const results = await session.run(feeds);
        
        // Estrazione dei risultati (il nome 'output' deve corrispondere all'export ONNX)
        const output = results.output.data;
        const maxIdx = argmax(output);
        
        // Visualizzazione del nome della specie
        const specieIdentificata = fishClasses[maxIdx] || "Specie non riconosciuta (ID: " + maxIdx + ")";
        predictionDiv.innerText = specieIdentificata;
        
        console.log("Risultato analisi:", specieIdentificata);
    } catch (e) {
        console.error("Errore durante la predizione:", e);
        predictionDiv.innerText = "Errore durante l'analisi dell'immagine.";
    }
});

/**
 * 4. Trasforma l'immagine in un Tensor Float32 [1, 3, 72, 256]
 */
async function preprocess(imgElement) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Dimensioni richieste dal tuo modello specifico
    const targetWidth = 256;
    const targetHeight = 72;
    
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    
    // Disegna l'immagine ridimensionandola nel canvas
    ctx.drawImage(imgElement, 0, 0, targetWidth, targetHeight);
    
    // Estrae i dati dei pixel (RGBA)
    const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight).data;

    // Normalizzazione e separazione dei canali (Format: NCHW -> 1, 3, 72, 256)
    const red = new Float32Array(targetWidth * targetHeight);
    const green = new Float32Array(targetWidth * targetHeight);
    const blue = new Float32Array(targetWidth * targetHeight);

    for (let i = 0, j = 0; i < imageData.length; i += 4, j++) {
        red[j] = imageData[i] / 255.0;     // R
        green[j] = imageData[i + 1] / 255.0; // G
        blue[j] = imageData[i + 2] / 255.0; // B
        // imageData[i + 3] è il canale Alpha, che scartiamo
    }

    // Combina i canali in un unico array Float32
    const combinedData = new Float32Array([...red, ...green, ...blue]);
    
    // Crea il Tensor finale
    return new ort.Tensor('float32', combinedData, [1, 3, targetHeight, targetWidth]);
}

/**
 * Trova l'indice del valore massimo in un array
 */
function argmax(array) {
    return array.indexOf(Math.max(...array));
}

// Avvia il caricamento
init();