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

def dense_layer(x, growth_rate):
    # BatchNorm -> ReLU -> 1x1 conv -> BatchNorm -> ReLU -> 3x3 conv
    y = layers.BatchNormalization()(x)
    y = layers.ReLU()(y)
    y = layers.Conv2D(4 * growth_rate, kernel_size=1,
                      padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(growth_rate, kernel_size=3,
                      padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(y)
    # Concatenate input and new features
    x = layers.Concatenate()([x, y])
    return x

def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        x = dense_layer(x, growth_rate)
    return x

def transition_layer(x, compression=0.5):
    # BN -> ReLU -> 1x1 conv -> AvgPool(2x2)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    in_channels = K.int_shape(x)[-1]         
    filters = int(in_channels * compression)  

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    return x

def build_model(input_shape=(224, 224, 3),
                   num_classes=8,
                   growth_rate=32,
                   block_layers=(6, 12, 24, 16),
                   compression=0.5):
    
    data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1)
    ])
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # Initial conv + max pool 
    x = layers.Conv2D(64, kernel_size=7, strides=2,
                      padding="same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Dense blocks + transition layers
    for i, num_layers in enumerate(block_layers):
        x = dense_block(x, num_layers, growth_rate)          
        if i != len(block_layers) - 1:
            x = transition_layer(x, compression=compression) 

    # Final BN -> ReLU -> global avg pool -> classifier
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

#model = build_model(
#    input_shape=(256, 256, 3), # Ensure this matches your image size
#    num_classes=50,
#    growth_rate=12, 
#    block_layers=(4, 8, 12, 8),
#    compression=0.5
#)

model = build_model(
    input_shape=(256, 256, 3), # Ensure this matches your image size
    num_classes=12,
    growth_rate=20, 
    block_layers=(6, 12, 24, 16),
    compression=0.5
)

print(model.summary())

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

checkpoint_path = "checkpoints/model_step_{batch:06d}.keras"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,  # Save full model (architecture + weights)
    save_freq='epoch',  
    verbose=1                 # Print a message when saving
)

# --- 4. Compile and Train ---
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_dataset, 
    epochs=300,
    validation_data=val_dataset, callbacks=[early_stopping_callback, lr_reducer, checkpoint_callback]
)

# Save the model
model.save('geo_model.keras')

