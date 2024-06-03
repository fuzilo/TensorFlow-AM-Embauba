import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import KFold
import os

data_dir = 'img'

# Definir parâmetros
batch_size = 32
img_height = 170
img_width = 170
num_classes = len(os.listdir(data_dir))  # Número de classes baseado nos subdiretórios
num_folds = 10
epochs = 100


# Função para criar o modelo
def create_model():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(img_height, img_width, 3)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Função para carregar o dataset de uma lista de arquivos
def load_data(file_paths, labels, img_height, img_width):
    images = []
    for file_path in file_paths:
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
    return np.array(images), np.array(labels)

# Preparar os caminhos e labels

file_paths = []
labels = []
class_names = os.listdir(data_dir)
class_indices = {class_name: i for i, class_name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    class_files = os.listdir(class_dir)
    file_paths.extend([os.path.join(class_dir, f) for f in class_files])
    labels.extend([class_indices[class_name]] * len(class_files))

# Inicializar KFold
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Para armazenar as acurácias de cada fold
fold_accuracies = []

# Validação cruzada
fold_no = 1
for train_index, val_index in kf.split(file_paths):
    print(f'Training fold {fold_no}...')
    
    train_files = [file_paths[i] for i in train_index]
    val_files = [file_paths[i] for i in val_index]
    train_labels = [labels[i] for i in train_index]
    val_labels = [labels[i] for i in val_index]
    
    x_train, y_train = load_data(train_files, train_labels, img_height, img_width)
    x_val, y_val = load_data(val_files, val_labels, img_height, img_width)
    
    # Normalizar os dados
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    
    model = create_model()
    
    # Treinar o modelo
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    
    # Avaliar o modelo
    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f'Fold {fold_no} validation accuracy: {val_acc}')
    
    fold_accuracies.append(val_acc)
    fold_no += 1

# Resultados da validação cruzada
print(f'Validation accuracies for each fold: {fold_accuracies}')
print(f'Mean validation accuracy: {np.mean(fold_accuracies)}')
print(f'Numero de classes: {num_classes}')

# Plotar a precisão de treinamento e validação ao longo das épocas
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()