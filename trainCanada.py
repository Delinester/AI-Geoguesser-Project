import tensorflow as tf
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import f1_score
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K, saving
import matplotlib.pyplot as plt
import os

# --- Configuration ---
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
BUFFER_SIZE = 1000
VALIDATION_SPLIT = 0.1
img_height = 256
img_width = 256
batch_size = 32
weight_decay = 1e-4

kmeans_model = None

directory = 'Remaining'

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='training',
    pad_to_aspect_ratio=True,
    verbose=True
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='validation',
    pad_to_aspect_ratio=True,
    verbose=True
)

base_model = tf.keras.models.load_model("US_classifier.keras")
base_model = tf.keras.models.load_model("US_classifier.keras")

print("Total layers in loaded model:", len(base_model.layers))

num_frozen = 200

for layer in base_model.layers[:num_frozen]:
    layer.trainable = False

for layer in base_model.layers[num_frozen:]:
    layer.trainable = True

x = base_model.layers[-2].output
new_logits = layers.Dense(
    2,                   # <- two output classes
    activation=None,     
    name="new_logits"
)(x)

model = models.Model(inputs=base_model.input, outputs=new_logits)

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"Trainable layers after freezing first {num_frozen}: {trainable_count}")

early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    min_delta=0.01,
    verbose=1,
    mode='auto',
    restore_best_weights=True
)

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=4,
    min_lr=1e-12,
    verbose=1,
)

checkpoint_path = "checkpoints_phoenix/model_step_{epoch:06d}.keras"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_freq='epoch',
    verbose=1
)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[early_stopping_callback, lr_reducer, checkpoint_callback]
)

model.save('geo_model_canada_2class.keras')