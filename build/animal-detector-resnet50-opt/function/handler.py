import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import json
import requests
import cgi
from io import BytesIO
import os

# -------------------------------
# Controlla dove si trova lo script
if '__file__' in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #cartella dove si trova lo script
else:
    SCRIPT_DIR = os.getcwd()

MODEL_PATH = os.path.join(SCRIPT_DIR, "resnet50.pth")

# -------------------------------
def ensure_model():
    """
    Controlla se il file modello esiste:
    - se sì, ritorna messaggio di conferma;
    - se no, scarica e salva il modello e ritorna messaggio.
    """
    if os.path.exists(MODEL_PATH):
        return f"File modello già presente: {MODEL_PATH}"
    else:
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        torch.save(model.state_dict(), MODEL_PATH)
        return f"File modello NON trovato. Scaricato e salvato in {MODEL_PATH}"

def load_model():
    """
    Carica il modello dai pesi salvati.
    Restituisce il modello e la trasformazione associata.
    """
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model, weights.transforms()

# -------------------------------
# Scarica le etichette ImageNet
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS = requests.get(LABELS_URL).text.strip().split("\n")

# -------------------------------
model_status_msg = ensure_model()
model, transform = load_model()

# -------------------------------
def handle(event, context):
    try:
        #Controlla se il file del modello esiste sul disco, cioè se resnet50.pth è presente nel percorso indicato da MODEL_PATH.
        model_exists = os.path.exists(MODEL_PATH)
        model_status_msg_local = f"File modello {'esiste' if model_exists else 'NON esiste'}: {MODEL_PATH}"

        body = event.body
        if isinstance(body, str):
            body = body.encode('utf-8')

        headers = dict(event.headers)
        content_type = headers.get("content-type") or headers.get("Content-Type", "")
        content_length = str(len(body))

        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": content_length
        }

        fs = cgi.FieldStorage(
            fp=BytesIO(body),
            environ=environ,
            keep_blank_values=True
        )

        if 'file' not in fs:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": "Nessun file trovato nella richiesta.",
                    "model_status": model_status_msg_local
                }),
                "headers": {"Content-Type": "application/json"}
            }

        file_item = fs['file']
        image = Image.open(file_item.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top3_probs, top3_classes = torch.topk(probabilities, 3)

        top3_results = []
        for i in range(3):
            class_idx = top3_classes[0][i].item()
            class_name = LABELS[class_idx]
            prob = top3_probs[0][i].item()
            top3_results.append({"class": class_name, "probability": prob})

        return {
            "statusCode": 200,
            "body": json.dumps({
                "top_3_predictions": top3_results,
                "model_status": model_status_msg_local
            }),
            "headers": {"Content-Type": "application/json"}
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": f"Errore durante l'elaborazione: {str(e)}",
                "model_status": model_status_msg_local
            }),
            "headers": {"Content-Type": "application/json"}
        }

