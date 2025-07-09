# 🐾 Animal Detector OPT (ResNet50)

Questa funzione serverless utilizza **ResNet50**, un modello pre-addestrato su **ImageNet**, per classificare immagini. Restituisce le **prime 3 predizioni** con le rispettive probabilità e gestisce automaticamente il download/salvataggio del modello in locale (`resnet50.pth`), ottimizzando così le prestazioni.

---

## 📦 Funzionalità

- Accetta una richiesta `POST` con `multipart/form-data` contenente un'immagine (`file`)
- Usa `torchvision.models.resnet50` con i pesi `IMAGENET1K_V1`
- Restituisce le **3 classi più probabili** con probabilità associate
- Salva il modello in locale come `resnet50.pth` alla prima esecuzione
- Controlla se il modello è già presente prima di eseguire la classificazione
- Supporta ambienti **serverless** come **OpenFaaS**, **AWS Lambda**, ecc.

---

## 🧠 Esempio di risposta JSON

```json
{
  "top_3_predictions": [
    {"class": "golden retriever", "probability": 0.87},
    {"class": "Labrador retriever", "probability": 0.08},
    {"class": "cocker spaniel", "probability": 0.03}
  ],
  "model_status": "File modello esiste: /path/to/resnet50.pth"
}

```

## ⌨️​ Esempio di input

curl -X POST http://INDIRIZZO/function/animal-detector-resnet50-opt  -H "Content-Type: multipart/form-data"   -F "file=@/home/kevin/Immagini/cane.jpeg"
Occorre prima salvare un'immagine e successivamente specificare il percorso appropriato!
