from pipeline import *
from losses import *

fold = 4
epochs = 300
batch_size = 64
# lr = 0.001

pipeline = Pipeline(train_fold='../folds/train_fold_{}.csv'.format(fold), valid_fold='../folds/valid_fold_{}.csv'.format(fold))

for lr in np.linspace(0.0009, 0.0001, 9):
    pipeline.training_stage_2(epochs=epochs, batch_size=batch_size, fold=fold, stage=2, lr=lr, loss='dice_loss',
                              custom_objects={'my_iou_metric': my_iou_metric, 'dice_loss': dice_loss}, weights='../stage_01/fold_04/weights/weights_stage_01_loss_dice_loss.model')


