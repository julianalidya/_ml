import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'dogs-vs-cats/train'

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=42
)

class_names = train_ds.class_names
print("Classes:", class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def build_model(version='v1'):
    layers = []

    if version in ['v2', 'v3']:
        layers.append(data_augmentation)

    layers += [
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
    ]

    if version == 'v3':
        layers.append(tf.keras.layers.Dropout(0.5))

    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

histories = {}

for version in ['v1', 'v2', 'v3']:
    print(f"\nTraining {version.upper()}...")
    model = build_model(version)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
    histories[version] = history
    val_acc = round(history.history['val_accuracy'][-1] * 100, 2)
    print(f"{version.upper()} Final Val Accuracy: {val_acc}%")
    model.save(f'model_{version}.keras')

plt.figure(figsize=(10, 5))
labels = {
    'v1': 'V1 - Basic CNN',
    'v2': 'V2 - + Augmentation',
    'v3': 'V3 - + Dropout'
}
for v, h in histories.items():
    plt.plot(h.history['val_accuracy'], label=labels[v])

plt.title('Validation Accuracy — All Versions')
plt.xlabel('Epoch')
plt.ylabel('Val Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('comparison_plot.png')
plt.show()

print('\n' + '='*45)
print(f'{"Version":<10} {"Val Accuracy":>15}  Notes')
print('='*45)
notes = {'v1': 'Basic CNN', 'v2': '+ Augmentation', 'v3': '+ Dropout'}
for v, h in histories.items():
    acc = str(round(h.history['val_accuracy'][-1]*100, 1)) + '%'
    print(f'{v.upper():<10} {acc:>15}  {notes[v]}')
print('='*45)
