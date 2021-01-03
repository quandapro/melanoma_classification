import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import Sequence

class DataLoader(Sequence):
    def __init__(self, image_ids, targets, image_folder, batch_size, augment = lambda x : x):
        assert len(image_ids) == len(targets), "Number of targets does not match number of image ids"
        self.image_ids = image_ids
        self.targets = targets 
        self.image_folder = image_folder
        self.batch_size = batch_size 
        self.indices = np.arange(len(image_ids))
        self.augment = augment

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def read_image(self, image_id):
        img = cv2.imread(os.path.join(self.image_folder, image_id + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.augment(img)
        return img

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        image_ids = self.image_ids[indices]
        targets = self.targets[indices]
        images = np.asarray([self.read_image(image_id) for image_id in image_ids], dtype='float32') / 255.
        return images, targets.astype('float32')
