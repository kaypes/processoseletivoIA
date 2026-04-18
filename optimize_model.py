import tensorflow as tf
import os

model = tf.keras.models.load_model("model.h5")
print("Modelo carregado: model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

original_size = os.path.getsize("model.h5")
optimized_size = os.path.getsize("model.tflite")

print(f"Modelo otimizado salvo em: model.tflite")
print(f"Tamanho original  (.h5):    {original_size / 1024:.1f} KB")
print(f"Tamanho otimizado (.tflite): {optimized_size / 1024:.1f} KB")
print(f"Redução: {(1 - optimized_size / original_size) * 100:.1f}%")
