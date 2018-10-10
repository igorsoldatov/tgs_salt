from pipeline import *
from losses import *

# pipeline = Pipeline()
# pipeline.make_folds()

fold = 4
epochs = 2
batch_size = 64
lr = 0.001

pipeline = Pipeline(train_fold='../folds/train_fold_{}.csv'.format(fold), valid_fold='../folds/valid_fold_{}.csv'.format(fold))

# pipeline.training_stage_1(epochs=epochs, batch_size=batch_size, fold=fold, lr=lr, loss='lovasz_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=batch_size, fold=fold, lr=lr, loss='dice_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=batch_size, fold=fold, lr=lr, loss='bce_dice_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=batch_size, fold=fold, lr=lr, loss='bce_logdice_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=batch_size, fold=fold, lr=lr, loss='weighted_bce_dice_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=batch_size, fold=fold, lr=lr, loss='binary_crossentropy')



