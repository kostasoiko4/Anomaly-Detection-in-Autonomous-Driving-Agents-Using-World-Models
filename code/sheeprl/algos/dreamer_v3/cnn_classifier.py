import os
#os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import cv2
import sys

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow import constant, float32
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import torch

#system 1 detector (paper Sec. IV-C): a compact cnn binary classifier over the paired real / imagined 24x24
#class index maps, trained with binary cross entropy. the class name is historical, the network is a classifier
class AnomalyAutoencoder:
    #loads the checkpoint if it exists, otherwise builds an untrained network.
    #threshold: sigmoid output at or above it -> 1 (anomaly), below it -> 0, see predict()
    def __init__(self, model_path=None, threshold=0.000001):
        #torch.cuda.empty_cache()
        K.clear_session()
        self.model = None
        self.threshold = threshold
        
        base_path = "/home/konstaoi/Desktop/konstaoi/Dreamer/sheeprl"
        
        #load the given checkpoint (one per obstacle regime), fall back to the default one
        if model_path is None:
            model_path = f"{base_path}/models/classifier/model_checkpoint_1.h5"

        if os.path.exists(model_path):
            self.model = load_model(model_path)
            self.build()
        else:
            self.build()
        
        #self.build()
        self.model.summary()
                
    #two conv blocks (64x5x5, 64x3x3, each with batch norm and 2x2 max pooling), dropout, 128 dense, sigmoid output
    def build(self):
        if self.model is None:
            print("Building CNN...")
            
            self.model = Sequential()
            
            self.model.add(Conv2D(64, (5,5), input_shape = (24, 24, 2), activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            
            self.model.add(Conv2D(64, (3,3), activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            
            self.model.add(Dropout(0.5))
            self.model.add(Flatten())
            
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            
            self.model.add(Dense(1, activation='sigmoid'))
        
            self.model.compile(Adam(lr = 0.001), loss="binary_crossentropy", metrics=['accuracy'])
    
    #fits the network on the train split, the validation split is only monitored
    def train(self, train_data, train_labels, val_data, val_labels):
        self.model.fit(train_data, train_labels, steps_per_epoch=10, epochs=800, validation_data=(val_data, val_labels), shuffle=1, verbose=2)
        
    #binary cross entropy and accuracy on the held out test split
    def evaluate(self, test_data, test_labels):
        score = self.model.evaluate(test_data, test_labels, verbose=0)
        print("===============================================")
        print(f"Test Score: {score[0]}")
        print(f"Test Accuracy: {score[1]}")
        
    #raw sigmoid output of one or more frame pairs, used by the threshold analysis in train_classifier.py
    def scores(self, data):
        return self.model.predict(data, verbose=0).ravel()

    #binary decision of one frame pair: sigmoid output >= threshold -> 1 (anomaly), else 0 (paper Sec. IV-C).
    #the threshold is set from the roc / pr analysis of the validation split (train_classifier.py THRESHOLD_ANALYSIS)
    def predict(self, data):
        pred = self.model.predict(data, verbose=0)
        if pred.item() < self.threshold:
            return 0
        else:
            return 1

    #saves the keras model to the given checkpoint path
    def save(self, model_path):
        if self.model is not None:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            print("model saved to", model_path)
    
