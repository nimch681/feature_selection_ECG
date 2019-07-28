import keras
from keras.callbacks import Callback

import math

class LossHistory(Callback):
    global step_decay

    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 14.0
        lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return lrate

    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []

    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.losses)))