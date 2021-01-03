from albumentations import *
import config 
from data_loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
import tensorflow_addons as tfa


def get_model():
    M = config.M
    base_model = M(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    out = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    model = Model(inputs=[base_model.input], outputs=[out])
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    loss = tfa.losses.SigmoidFocalCrossEntropy(gamma=2.0, alpha=0.9)
    model.compile(optimizer=opt, 
                loss=loss,
                metrics=[AUC()])
    return model

def train(model, train_dataloader, test_dataloader, callbacks=[]):
    return model.fit(train_dataloader,
                     epochs=config.num_of_epochs,
                     validation_data=test_dataloader,
                     callbacks=callbacks,
                     verbose=1)

def augment(image):
    aug = Compose([
        Flip(),
        ShiftScaleRotate(),
        Cutout(num_holes=4, max_h_size=config.img_size // 8, max_w_size=config.img_size // 8),
        RandomBrightnessContrast()
    ])
    return aug(image=image)['image']


if __name__ == '__main__':
    datacsv = pd.read_csv(config.datacsv)
    image_ids = np.asarray(datacsv['image_name'])
    targets = np.asarray(datacsv['target'])

    X_train, X_test, y_train, y_test = train_test_split(image_ids, targets, test_size=0.2, random_state=42) # Fixed random state for consistent validation split across all experiments
    train_dataloader = DataLoader(X_train, y_train, config.image_folder, config.batch_size, augment)
    test_dataloader = DataLoader(X_test, y_test, config.image_folder, config.batch_size)

    callbacks = [ModelCheckpoint(os.path.join(config.ckpt_folder, f'{config.model_name}.h5'),
                                verbose=1,
                                monitor='val_auc',
                                mode='max',
                                save_best_only=True,
                                save_weights_only=True)]
    model = get_model()
    hist = train(model, train_dataloader, test_dataloader, callbacks)
    val_AUC = max(hist.history['val_auc'])
    print(val_AUC)