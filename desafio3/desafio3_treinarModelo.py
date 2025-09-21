
#Bibliotecas
import kagglehub
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

#Baixar dataset
path = kagglehub.dataset_download("emmarex/plantdisease")
print("Path to dataset files:", path)

data_dir = os.path.join(path, "PlantVillage")
print("Diretório de dados:", data_dir)
print("Subpastas/classes encontradas:", os.listdir(data_dir)[:5])

# 3. Criar datasets (mesmo, IMG_SIZE=224)
IMG_SIZE = 160
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

ds_train = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=42
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=42
)

class_names = ds_train.class_names
num_classes = len(class_names)
print("Número de classes:", num_classes)
print("Primeiras classes:", class_names[:5])

#Otimizar datasets
AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)

#Modelo com Transfer Learning
#Base pré-treinada
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

#Congelar base inicialmente
base_model.trainable = False

#Augmentação
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])

# Modelo completo
model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

#Compilação inicia
initial_learning_rate = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#Callback pra reduzir LR
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)

# Treino em 2 fases
history1 = model.fit(
    ds_train, validation_data=ds_val, epochs=4  # Fase 1: Top layers
)

# Descongelar base pra fine-tuning
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate / 10),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    ds_train, validation_data=ds_val, epochs=3,
    callbacks=[lr_scheduler]
)

# Juntar histories
history = {}
for k in history1.history:
    history[k] = history1.history[k] + history2.history[k]


#Avaliar
loss, acc = model.evaluate(ds_val)
print(f"Acurácia no conjunto de teste: {acc:.2f}")


#Plotar curvas

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Treino")
plt.plot(history["val_accuracy"], label="Validação")
plt.title("Evolução da Acurácia")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Treino")
plt.plot(history["val_loss"], label="Validação")
plt.title("Evolução da Perda")
plt.xlabel("Épocas")
plt.ylabel("Perda")
plt.legend()

plt.tight_layout()
plt.show()


# 7. Teste de previsão

for image_batch, label_batch in ds_val.take(1):
    first_image = image_batch[0]
    first_label = int(label_batch[0])

    pred = model.predict(tf.expand_dims(first_image, 0))
    predicted_class = np.argmax(pred[0])
    confidence = np.max(pred[0]) * 100

    # Top-3
    top3_idx = np.argsort(pred[0])[::-1][:3]
    top3_classes = [class_names[i] for i in top3_idx]
    top3_conf = pred[0][top3_idx] * 100

    plt.figure(figsize=(8, 6))
    plt.imshow(first_image)
    plt.title(f"Real: {class_names[first_label]}\nPrevisto: {class_names[predicted_class]} ({confidence:.2f}%)\nTop-3: {top3_classes[0]} ({top3_conf[0]:.1f}%), {top3_classes[1]} ({top3_conf[1]:.1f}%), {top3_classes[2]} ({top3_conf[2]:.1f}%)")
    plt.axis("off")
    plt.show()

    print("Classe real:", class_names[first_label])
    print("Classe prevista:", class_names[predicted_class])
    print("Confiança:", f"{confidence:.2f}%")

# Salvar
model.save("modelo_plantvillage_transfer.keras")
print("Modelo salvo como modelo_plantvillage_transfer.keras!")

from google.colab import files
files.download("modelo_plantvillage_transfer.keras")