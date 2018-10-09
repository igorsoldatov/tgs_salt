from pipeline import *
from losses import *

pipeline = Pipeline()
pipeline.make_folds()

fold = 4
epochs = 200

pipeline = Pipeline(train_fold='../folds/train_fold_{}.csv'.format(fold), valid_fold='../folds/valid_fold_{}.csv'.format(fold))

pipeline.training_stage_1(epochs=epochs, batch_size=32, fold=fold, lr=0.001, loss='dice_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=32, fold=fold, lr=0.001, loss='bce_dice_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=32, fold=fold, lr=0.001, loss='bce_logdice_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=32, fold=fold, lr=0.001, loss='weighted_bce_dice_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=32, fold=fold, lr=0.001, loss='lovasz_loss')
pipeline.training_stage_1(epochs=epochs, batch_size=32, fold=fold, lr=0.001, loss='binary_crossentropy')

# for lr in np.linspace(0.001, 0.0001, 10):
#     pipeline.training_stage_1(epochs=100, batch_size=32, fold=fold, lr=lr)


