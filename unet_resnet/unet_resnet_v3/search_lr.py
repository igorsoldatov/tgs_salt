from pipeline import *
from losses import *

pipeline = Pipeline()
pipeline.make_folds()

fold = 4

pipeline = Pipeline(train_fold='../folds/train_fold_{}.csv'.format(fold), valid_fold='../folds/valid_fold_{}.csv'.format(fold))

for lr in np.linspace(0.001, 0.0001, 10):
    pipeline.training_stage_1(epochs=100, batch_size=32, fold=fold, lr=lr)


