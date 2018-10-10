from pipeline import *
from losses import *

fold = 4
epochs = 100
batch_size = 64
lr = 0.001

pipeline = Pipeline(train_fold='../folds/train_fold_{}.csv'.format(fold), valid_fold='../folds/valid_fold_{}.csv'.format(fold))

for lr in np.linspace(0.001, 0.0001, 10):
    pipeline.training_stage_1(epochs=epochs, batch_size=batch_size, fold=fold, lr=lr)


