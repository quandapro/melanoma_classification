from tensorflow.keras.applications import EfficientNetB0

batch_size = 16
ckpt_folder = './ckpt'
datacsv = './data/train.csv'
image_folder = './data/train_preprocessed'
img_size = 256
M = EfficientNetB0
model_name = M.__name__
num_of_epochs = 30




