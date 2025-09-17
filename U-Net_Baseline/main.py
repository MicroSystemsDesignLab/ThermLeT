#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras import Model, regularizers
import numpy as np
import matplotlib.pyplot as plt

#Hyperparameters
reg_rate           = 0.001
initial_learning_rate = 0.001
batch_size         = 1
epochs             = 50
target_size        = (224, 224)

#    Data Loading & Normalization
# Load full training arrays (shape: N × 34 × 32)
power_map_train = np.load('power_map_train.npy').astype(np.float32)
temp_map_train  = np.load('temp_map_train.npy').astype(np.float32)

# Add channel dimension → (N, 34, 32, 1)
power_map_train = power_map_train[..., np.newaxis]
temp_map_train  = temp_map_train[...,  np.newaxis]


# Compute global min/max over entire training set
p_min, p_max = power_map_train.min(), power_map_train.max()
t_min, t_max = temp_map_train.min(),  temp_map_train.max()

# Build tf.data pipeline: resize → normalize → batch → prefetch

train_dataset = (
    tf.data.Dataset
      .from_tensor_slices((power_map_train, temp_map_train))
      .map(lambda x, y: (
          tf.image.resize(x, target_size),
          tf.image.resize(y, target_size)
      ), num_parallel_calls=tf.data.AUTOTUNE)
      .map(lambda x, y: (
          # min–max normalize to [0,1]
          (x - p_min) / (p_max - p_min),
          (y - t_min) / (t_max - t_min)
      ), num_parallel_calls=tf.data.AUTOTUNE)
      .batch(batch_size)
      .prefetch(tf.data.AUTOTUNE)
)
l2 = regularizers.l2

class Encoder(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(64, 3, activation='relu', padding='same',
                            kernel_regularizer=l2(reg_rate),
                            bias_regularizer=l2(reg_rate))
        self.pool1 = MaxPooling2D(2, padding='same')
        self.conv2 = Conv2D(32, 3, activation='relu', padding='same',
                            kernel_regularizer=l2(reg_rate),
                            bias_regularizer=l2(reg_rate))
        self.pool2 = MaxPooling2D(2, padding='same')
        self.conv3 = Conv2D(16, 3, activation='relu', padding='same',
                            kernel_regularizer=l2(reg_rate),
                            bias_regularizer=l2(reg_rate))
        self.pool3 = MaxPooling2D(2, padding='same')

    def call(self, x):
        x0 = self.conv1(x)    # -> [B,224,224,64]
        x1 = self.pool1(x0)   # -> [B,112,112,64]
        x1 = self.conv2(x1)   # -> [B,112,112,32]
        x2 = self.pool2(x1)   # -> [B, 56, 56,32]
        x2 = self.conv3(x2)   # -> [B, 56, 56,16]
        x3 = self.pool3(x2)   # -> [B, 28, 28,16]
        return x0, x1, x2, x3

class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.up1        = UpSampling2D(2)
        self.deconv1    = Conv2DTranspose(16, 3, activation='relu', padding='same',
                                          kernel_regularizer=l2(reg_rate),
                                          bias_regularizer=l2(reg_rate))
        self.up2        = UpSampling2D(2)
        self.deconv2    = Conv2DTranspose(32, 3, activation='relu', padding='same',
                                          kernel_regularizer=l2(reg_rate),
                                          bias_regularizer=l2(reg_rate))
        self.up3        = UpSampling2D(2)
        self.deconv3    = Conv2DTranspose(64, 3, activation='relu', padding='same',
                                          kernel_regularizer=l2(reg_rate),
                                          bias_regularizer=l2(reg_rate))
        self.final_conv = Conv2DTranspose(1, 3, activation='linear', padding='same',
                                          kernel_regularizer=l2(reg_rate),
                                          bias_regularizer=l2(reg_rate))

    def call(self, feats):
        x0, x1, x2, x3 = feats
        x = self.up1(x3)                       # [B,56,56,16]
        x = tf.concat([x, x2], axis=-1)       # [B,56,56,32]
        x = self.deconv1(x)                   # [B,56,56,16]
        x = self.up2(x)                       # [B,112,112,16]
        x = tf.concat([x, x1], axis=-1)       # [B,112,112,32]
        x = self.deconv2(x)                   # [B,112,112,32]
        x = self.up3(x)                       # [B,224,224,32]
        x = tf.concat([x, x0], axis=-1)       # [B,224,224,96]
        x = self.deconv3(x)                   # [B,224,224,64]
        return self.final_conv(x)             # [B,224,224,1]

class Autoencoder(Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x):
        feats = self.encoder(x)
        return self.decoder(feats)

model = Autoencoder()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.98, staircase=True
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='mse',
    metrics=['mse','mae','mape']
)

# Training
history = model.fit(train_dataset, epochs=epochs)

for x_batch, y_true_batch in train_dataset.take(1):
    y_pred_batch = model.predict(x_batch)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Input Power Map")
    plt.imshow(x_batch[0,...,0], cmap='magma')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.title("True Temp")
    plt.imshow(y_true_batch[0,...,0], cmap='hot')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.title("Predicted Temp")
    plt.imshow(y_pred_batch[0,...,0], cmap='hot')
    plt.colorbar()
    plt.show()
    break
