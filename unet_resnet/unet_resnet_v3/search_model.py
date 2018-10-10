from model_unet_resnet34 import *
from pipeline import *
from losses import *

fold = 4
epochs = 100
batch_size = 64
lr = 0.001

pipeline = Pipeline(img_size_target=128, train_fold=f'../folds/train_fold_{fold}.csv', valid_fold=f'../folds/valid_fold_{fold}.csv')

pipeline.training_unet_resnet34(epochs=epochs, batch_size=batch_size, fold=fold, lr=lr, loss='binary_crossentropy')
