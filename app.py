import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_ICONS = {
    "buildings": "🏢",
    "forest": "🌲",
    "glacier": "🧊",
    "mountain": "⛰️",
    "sea": "🌊",
    "street": "🛣️",
}
IMG_SIZE = 224

_PYTORCH = None
_TENSORFLOW = None


def load_pytorch_model():
    global _PYTORCH
    if _PYTORCH is None:
        import torch
        import torch.nn as nn

        class IntelCNNPyTorch(nn.Module):
            def __init__(self, num_classes=6):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = self.gap(x)
                return self.classifier(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IntelCNNPyTorch(num_classes=len(CLASSES)).to(device)
        model.load_state_dict(torch.load("maimouna_model.pth", map_location=device))
        model.eval()
        _PYTORCH = (model, device)
    return _PYTORCH


def load_tensorflow_model():
    global _TENSORFLOW
    if _TENSORFLOW is None:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        _TENSORFLOW = tf.keras.models.load_model(
            "maimouna_model.keras", compile=False
        )
    return _TENSORFLOW


def preprocess_pytorch(image):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def preprocess_tensorflow(image):
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        model_choice = request.form.get("model", "pytorch")

        if model_choice == "pytorch":
            import torch

            model, device = load_pytorch_model()
            tensor = preprocess_pytorch(image).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        elif model_choice == "tensorflow":
            model = load_tensorflow_model()
            arr = preprocess_tensorflow(image)
            probs = np.array(model.predict(arr, verbose=0)[0])
        else:
            return jsonify({"error": "Unknown model"}), 400

        idx = int(np.argmax(probs))
        predicted_class = CLASSES[idx]
        confidence = round(float(probs[idx]) * 100, 2)

        return jsonify(
            {
                "class": predicted_class,
                "icon": CLASS_ICONS[predicted_class],
                "confidence": confidence,
                "model": model_choice,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
