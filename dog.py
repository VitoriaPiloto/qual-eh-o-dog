import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def load_labels(label_path):
    """Carrega os rótulos do arquivo .txt com suporte a acentos"""
    with open(label_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def preprocess_image(image_path, input_size):
    """Redimensiona e normaliza a imagem para o modelo"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(input_size)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normaliza para [0, 1]
    return np.expand_dims(image_array, axis=0)


def predict_image(model_path, label_path, image_path):
    """Realiza a inferência em uma imagem usando o modelo TFLite"""
    labels = load_labels(label_path)
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = tuple(input_details[0]['shape'][1:3])

    input_data = preprocess_image(image_path, input_size)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(output_data[0])
    predicted_label = labels[predicted_index]
    confidence = output_data[0][predicted_index]

    return predicted_label, confidence


from tkinter import Scrollbar

def load_image():
    """Abre o seletor de arquivos para o usuário carregar uma imagem"""
    global img_path, img_display
    img_path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
    if img_path:
        image = Image.open(img_path)

        # Ajusta o tamanho da imagem mantendo a proporção
        max_size = (400, 400)  # Tamanho máximo para redimensionar
        image.thumbnail(max_size)

        img_display = ImageTk.PhotoImage(image)

        # Atualiza o Label de imagem com Scrollbar
        img_canvas.config(scrollregion=(0, 0, image.width, image.height))  # Ajusta o scroll para a imagem
        img_canvas.create_image(0, 0, anchor="nw", image=img_display)
        img_canvas.image = img_display

def run_prediction():
    """Executa a predição na imagem carregada"""
    if not img_path:
        messagebox.showerror("Erro", "Por favor, carregue uma imagem.")
        return

    try:
        predicted_label, confidence = predict_image(model_path, label_path, img_path)
        result_label.config(
            text=f"Predição: {predicted_label}\nConfiança: {confidence:.2f}",
            fg="green",
        )
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao fazer a predição.\n{e}")


# Configurações do modelo e rótulos
model_path = "modelo.tflite"
label_path = "labels.txt"
img_path = None


# Interface gráfica principal
root = tk.Tk()
root.title("Classificador de Doguinhos")
root.geometry("400x600")
root.resizable(False, False)
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, padx=10, pady=10, bg="#f0f0f0")
frame.pack()

# Título
title_label = tk.Label(frame, text="Qual é o dog?", font=("Arial", 18, "bold"), bg="#f0f0f0")
title_label.grid(row=0, column=0, pady=10)

# Botão para carregar imagem
load_btn = ttk.Button(frame, text="Carregar Imagem", command=load_image)
load_btn.grid(row=1, column=0, pady=10)

# Canvas com barras de rolagem para exibir a imagem
img_frame = tk.Frame(frame, bg="#f0f0f0")
img_frame.grid(row=2, column=0, pady=10)

# Canvas para exibir a imagem
img_canvas = tk.Canvas(img_frame, width=300, height=300, bg="#f0f0f0")
img_canvas.grid(row=0, column=0)

# Barras de rolagem
scroll_x = Scrollbar(img_frame, orient="horizontal", command=img_canvas.xview)
scroll_x.grid(row=1, column=0, sticky="ew")

scroll_y = Scrollbar(img_frame, orient="vertical", command=img_canvas.yview)
scroll_y.grid(row=0, column=1, sticky="ns")

img_canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)

# Botão para executar a predição
predict_btn = ttk.Button(frame, text="Diz aí, qual o dog?", command=run_prediction)
predict_btn.grid(row=3, column=0, pady=10)

# Label para exibir o resultado
result_label = tk.Label(
    frame,
    text="Resultado da predição aparecerá aqui.",
    wraplength=300,
    justify="center",
    font=("Arial", 12),
    bg="#f0f0f0",
    fg="#333",
)
result_label.grid(row=4, column=0, pady=20)

# Inicia a aplicação
root.mainloop()