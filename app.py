import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageTk

model = tf.keras.models.load_model("mnist_cnn.keras")

window = tk.Tk()
window.title("MNIST Digit Recognizer")
window.geometry("620x700")
window.resizable(False, False)

CANVAS_SIZE = 400

main_frame = tk.Frame(window)
main_frame.pack(pady=15)

canvas = tk.Canvas(
    main_frame,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
    highlightthickness=0
)
canvas.grid(row=0, column=0, padx=15)

preview_frame = tk.Frame(main_frame)
preview_frame.grid(row=0, column=1, padx=15)

preview_title = tk.Label(
    preview_frame,
    text="28×28 Input",
    font=("Arial", 14)
)
preview_title.pack()

preview_canvas = tk.Canvas(
    preview_frame,
    width=140,
    height=140,
    bg="black",
    highlightthickness=1
)
preview_canvas.pack(pady=10)

image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    0
)

draw = ImageDraw.Draw(image)

last_x = None
last_y = None


def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y


def draw_digit(event):
    global last_x, last_y

    x = event.x
    y = event.y

    canvas.create_line(
        last_x,
        last_y,
        x,
        y,
        fill="white",
        width=16,
        capstyle=tk.ROUND,
        smooth=True
    )

    draw.line(
        (last_x, last_y, x, y),
        fill=255,
        width=16
    )

    r = 8

    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        fill=255
    )

    last_x = x
    last_y = y


def stop_draw(event):
    global last_x, last_y
    last_x = None
    last_y = None


canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", stop_draw)


def preprocess_image():

    img = np.array(image)

    rows, cols = np.where(img > 20)

    if len(rows) == 0:
        return None

    top = rows.min()
    bottom = rows.max()
    left = cols.min()
    right = cols.max()

    cropped = image.crop(
        (left, top, right + 1, bottom + 1)
    )

    width, height = cropped.size
    size = max(width, height)

    square = Image.new(
        "L",
        (size, size),
        0
    )

    x_offset = (size - width) // 2
    y_offset = (size - height) // 2

    square.paste(
        cropped,
        (x_offset, y_offset)
    )

    square = square.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    final_image.paste(
        square,
        (4, 4)
    )

    return final_image


def update_preview(processed):

    preview = processed.resize(
        (140, 140),
        Image.Resampling.NEAREST
    )

    preview_image = ImageTk.PhotoImage(preview)

    preview_canvas.delete("all")

    preview_canvas.create_image(
        70,
        70,
        image=preview_image
    )

    preview_canvas.image = preview_image


def predict_digit():

    processed = preprocess_image()

    if processed is None:
        result_label.config(
            text="Draw a digit first!"
        )
        return

    update_preview(processed)

    img = np.array(processed)

    img = img.astype("float32") / 255.0

    img = img.reshape(
        1,
        28,
        28,
        1
    )

    prediction = model.predict(
        img,
        verbose=0
    )[0]

    digit = np.argmax(prediction)
    confidence = prediction[digit] * 100

    result_label.config(
        text=f"Prediction: {digit}   Confidence: {confidence:.2f}%"
    )

    for i in range(10):

        percentage = prediction[i] * 100

        bar, percentage_label = confidence_bars[i]

        bar.delete("all")

        bar.create_rectangle(
            0,
            0,
            min(200, percentage * 2),
            15,
            fill="blue",
            outline=""
        )

        percentage_label.config(
            text=f"{percentage:.2f}%"
        )


def clear_canvas():

    canvas.delete("all")

    draw.rectangle(
        (0, 0, CANVAS_SIZE, CANVAS_SIZE),
        fill=0
    )

    preview_canvas.delete("all")

    result_label.config(
        text="Draw a digit"
    )

    for bar, percentage_label in confidence_bars:

        bar.delete("all")

        percentage_label.config(
            text="0.00%"
        )


button_frame = tk.Frame(window)
button_frame.pack(pady=5)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    width=14,
    height=2
)

predict_button.grid(
    row=0,
    column=0,
    padx=5
)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    width=14,
    height=2
)

clear_button.grid(
    row=0,
    column=1,
    padx=5
)

result_label = tk.Label(
    window,
    text="Draw a digit",
    font=("Arial", 18)
)

result_label.pack(pady=10)

confidence_frame = tk.Frame(window)
confidence_frame.pack(pady=5)

confidence_bars = []

for i in range(10):

    label = tk.Label(
        confidence_frame,
        text=f"{i}:",
        font=("Consolas", 11),
        width=3
    )

    label.grid(
        row=i,
        column=0
    )

    bar = tk.Canvas(
        confidence_frame,
        width=200,
        height=15,
        bg="lightgray",
        highlightthickness=0
    )

    bar.grid(
        row=i,
        column=1,
        padx=5
    )

    percentage_label = tk.Label(
        confidence_frame,
        text="0.00%",
        font=("Consolas", 10),
        width=7
    )

    percentage_label.grid(
        row=i,
        column=2
    )

    confidence_bars.append(
        (bar, percentage_label)
    )

window.mainloop()