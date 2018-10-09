from pipeline import *

# pipeline = Pipline()
# pipeline.make_folds()

fold = 3

pipeline = Pipline(train_fold='../folds/train_fold_{}.csv'.format(fold), valid_fold='../folds/valid_fold_{}.csv'.format(fold))

pipeline.training_stage_1(epochs=10, batch_size=32, fold=fold)

threshold_best, iou_best, iou = pipeline.validate(stage=1, fold=fold, custom_objects={'my_iou_metric': my_iou_metric})

# pipeline.training_stage_2(epochs=200, batch_size=32, fold=fold)
#
# threshold_best, iou_best, iou = pipeline.validate(stage=2, fold=fold, custom_objects={'my_iou_metric': my_iou_metric, 'lovasz_loss': lovasz_loss})
#
# pipeline.training_stage_3(epochs=50, batch_size=32, fold=0)
#
# threshold_best, iou_best, iou = pipeline.validate(stage=3, fold=fold, custom_objects={'my_iou_metric': my_iou_metric, 'lovasz_loss': lovasz_loss})

pipeline.predict(1, fold, 0.58)
# pipeline.predict(2, fold, 0.50)
# pipeline.predict(3, fold, 0.50)

