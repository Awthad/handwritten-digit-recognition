from flask import Flask, request, render_template_string
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = tf.keras.models.load_model("mnist_cnn.keras")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Digit Recognition</title>
</head>
<body>
    <h1>Handwritten Digit Recognition</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
        <h2>Prediction: {{ prediction }}</h2>
        <h3>Confidence: {{ confidence }}%</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("L")
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)

        probabilities = model.predict(image, verbose=0)[0]
        prediction = int(np.argmax(probabilities))
        confidence = round(float(np.max(probabilities)) * 100, 2)

    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence
    )

app.run(host="0.0.0.0", port=5000)