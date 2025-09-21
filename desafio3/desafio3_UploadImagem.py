import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from google.colab import files

try:
    import kagglehub
    dataset_root = "/content/PlantVillage"
    if not os.path.exists(dataset_root):
        path = kagglehub.dataset_download("emmarex/plantdisease")
        dataset_root = os.path.join(path, "PlantVillage")
        print("Dataset baixado em:", path)
    else:
        print("Dataset já presente em:", dataset_root)
except Exception as e:
    dataset_root = "/content/PlantVillage"
    print("kagglehub não disponível ou ocorreu erro (seguindo sem baixar).")

# Obtem class_names de forma segura (ordem alfabética → mesma do image_dataset_from_directory)
if os.path.exists(dataset_root):
    class_names = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    print(f"Detectadas {len(class_names)} classes a partir de {dataset_root}")
else:
    #fallback
    class_names = ['Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
                   'Blueberry_healthy', 'Cherry_(including_sour)_Powdery_mildew', 'Cherry_(including_sour)_healthy',
                   'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)_Common_rust_',
                   'Corn_(maize)_Northern_Leaf_Blight', 'Corn_(maize)_healthy', 'Grape_Black_rot',
                   'Grape_Esca_(Black_Measles)', 'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape_healthy',
                   'Orange_Haunglongbing_(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
                   'Pepper,_bell__Bacterial_spot', 'Pepper,_bell__healthy', 'Potato__Early_blight',
                   'Potato__Late_blight', 'Potato__healthy', 'Raspberry__healthy', 'Soybean__healthy',
                   'Squash__Powdery_mildew', 'Strawberry__Leaf_scorch', 'Strawberry__healthy',
                   'Tomato__Bacterial_spot', 'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Leaf_Mold',
                   'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites_Two-spotted_spider_mite',
                   'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                   'Tomato__healthy']
    print("Usando lista fallback de class_names (verifique se está na ORDEM certa).")

# Carrega modelo salvo
MODEL_PATH = "modelo_plantvillage_transfer.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Arquivo de modelo não encontrado em {MODEL_PATH}. Faça upload ou rode o treino primeiro.")
model = keras.models.load_model(MODEL_PATH)
print("Modelo carregado com sucesso:", MODEL_PATH)

#Detecta se o modelo já contém uma camada Rescaling(1./255)
def model_has_rescaling(m):
    def search(layer):
        if isinstance(layer, tf.keras.layers.Rescaling):
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if search(sub):
                    return True
        return False
    for layer in m.layers:
        if search(layer):
            return True
    return False

_has_rescaling = model_has_rescaling(model)
print("Modelo contém Rescaling(1./255)? ->", _has_rescaling)

#Upload imagem e predição
print("Selecione a(s) imagem(ns) para enviar (do seu PC).")
uploaded = files.upload()
if len(uploaded) == 0:
    raise RuntimeError("Nenhuma imagem enviada.")

for img_name in uploaded.keys():
    pil_img = image.load_img(img_name, target_size=(160,160))
    arr = image.img_to_array(pil_img).astype("float32")

    #Para o predict: se o modelo já tem Rescaling, envie 0..255; se não, normalize para 0..1
    if _has_rescaling:
        input_array = np.expand_dims(arr, axis=0)
    else:
        input_array = np.expand_dims(arr / 255.0, axis=0)

    # predição
    preds = model.predict(input_array, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds)) * 100.0

    #top 3
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [(class_names[i], preds[i]*100) for i in top3_idx]

    plt.figure(figsize=(6,6))
    plt.imshow(arr.astype("uint8"))
    plt.axis("off")
    plt.title(f"Previsto: {class_names[idx]} ({conf:.2f}%)")
    plt.show()

    print("Arquivo:", img_name)
    print("Classe prevista:", class_names[idx])
    print("Confiança:", f"{conf:.2f}%")
    print("Top-3:")
    for c,p in top3:
        print(f" - {c}: {p:.2f}%")
    print("-"*40)

print("Pronto — predições concluídas.")
