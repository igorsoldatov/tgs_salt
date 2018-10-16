from model_unet_resnet import *
from model_unet_resnet34 import *
from lib import *
from losses import *

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from os import makedirs, listdir
from os.path import isfile, join
from tqdm import tqdm_notebook, tqdm
from skimage.transform import resize

def get_files(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort(reverse=True)
    return files


class Pipeline:
    def __init__(self, img_size_target=101, train_fold='', valid_fold=''):
        self.t_start = time.time()
        self.version = 1
        self.cv_total = 5
        self.img_size_target = img_size_target
        self.img_size_ori = 101
        self.submission_file = '../stage_{}/fold_{}/submissions/sub_stage_{}_fold_{}_iou_thresh_{}.csv'
        self.prepare_data(train_fold, valid_fold)

    def prepare_data(self, train_fold='', valid_fold=''):
        # Loading of training/testing ids and depths
        if train_fold != '' and valid_fold != '':
            self.train_df = pd.read_csv(train_fold, index_col="id", usecols=[0])
            self.valid_df = pd.read_csv(valid_fold, index_col="id", usecols=[0])
            self.depths_df = pd.read_csv("../../input/depths.csv", index_col="id")
            self.train_df = self.train_df.join(self.depths_df)
            self.test_df = self.depths_df[
                ~self.depths_df.index.isin(self.train_df.index) & ~self.depths_df.index.isin(self.valid_df.index)]

            print(len(self.train_df))

            self.train_df["images"] = [
                np.array(load_img("../../input/train/images/{}.png".format(idx), color_mode='grayscale')) / 255 for n, idx
                in tqdm(enumerate(self.train_df.index), total=len(self.train_df.index))]
            self.train_df["masks"] = [
                np.array(load_img("../../input/train/masks/{}.png".format(idx), color_mode='grayscale')) / 255 for n, idx
                in tqdm(enumerate(self.train_df.index), total=len(self.train_df.index))]
            self.valid_df["images"] = [
                np.array(load_img("../../input/train/images/{}.png".format(idx), color_mode='grayscale')) / 255 for n, idx
                in tqdm(enumerate(self.valid_df.index), total=len(self.valid_df.index))]
            self.valid_df["masks"] = [
                np.array(load_img("../../input/train/masks/{}.png".format(idx), color_mode='grayscale')) / 255 for n, idx
                in tqdm(enumerate(self.valid_df.index), total=len(self.valid_df.index))]

            self.train_df["coverage"] = self.train_df.masks.map(np.sum) / pow(self.img_size_target, 2)
            self.train_df["coverage_class"] = self.train_df.masks.map(get_mask_type)
            self.valid_df["coverage"] = self.valid_df.masks.map(np.sum) / pow(self.img_size_target, 2)
            self.valid_df["coverage_class"] = self.valid_df.masks.map(get_mask_type)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.x_test = np.array(
                    [self.upsample(np.array(load_img('../../input/test/images/{}.png'.format(idx), color_mode='grayscale'))) / 255 for
                     n, idx in tqdm(enumerate(self.test_df.index), total=len(self.test_df.index))]).reshape(-1, self.img_size_target, self.img_size_target, 1)

        else:
            self.train_df = pd.read_csv("../../input/train.csv", index_col="id", usecols=[0])
            self.depths_df = pd.read_csv("../../input/depths.csv", index_col="id")
            self.train_df = self.train_df.join(self.depths_df)
            self.test_df = self.depths_df[~self.depths_df.index.isin(self.train_df.index)]

            print(len(self.train_df))

            self.train_df["images"] = [
                np.array(load_img("../../input/train/images/{}.png".format(idx), color_mode='grayscale')) / 255 for n, idx
                in tqdm(enumerate(self.train_df.index), total=len(self.train_df.index))]
            self.train_df["masks"] = [
                np.array(load_img("../../input/train/masks/{}.png".format(idx), color_mode='grayscale')) / 255 for n, idx
                in tqdm(enumerate(self.train_df.index), total=len(self.train_df.index))]

            self.train_df["coverage"] = self.train_df.masks.map(np.sum) / pow(self.img_size_target, 2)
            self.train_df["coverage_class"] = self.train_df.masks.map(get_mask_type)

            self.train_all = []
            self.evaluate_all = []
            skf = StratifiedKFold(n_splits=self.cv_total, random_state=1234, shuffle=True)
            for train_index, evaluate_index in skf.split(self.train_df.index.values, self.train_df.coverage_class):
                self.train_all.append(train_index)
                self.evaluate_all.append(evaluate_index)
                print(train_index.shape,
                      evaluate_index.shape)  # the shape is slightly different in different cv, it's OK

            self.x_test = np.array(
                [(np.array(load_img("../../input/test/images/{}.png".format(idx), color_mode='grayscale'))) / 255 for
                 idx in
                 self.test_df.index]).reshape(-1, self.img_size_target, self.img_size_target, 1)

    def training_unet_resnet34(self, epochs, batch_size=32, fold=0, lr=0.001, loss='binary_crossentropy'):
        # training
        stage = 1
        self.create_stage_paths(stage, fold)
        print(' ###################################################################################################\n',
              '####################################################################################################\n',
              'training {} stage, {} fold, {} epochs, {} batch_size\n'.format(stage, fold, epochs, batch_size),
              '{} lr, {}'.format(lr, loss))

        x_train, y_train, x_valid, y_valid = self.get_cv_data()

        # Data augmentation
        x_train_aug = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train_aug = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
        x_train_aug = np.repeat(x_train_aug, 3, axis=3)
        x_valid = np.repeat(x_valid, 3, axis=3)
        # x_train_aug = np.append(x_train_aug, [np.flipud(x) for x in x_train], axis=0)
        # y_train_aug = np.append(y_train_aug, [np.flipud(x) for x in y_train], axis=0)

        model = UResNet34(input_shape=(self.img_size_target, self.img_size_target, 3))

        model.summary()

        c = optimizers.adam(lr=lr)
        model.compile(loss=get_loss_by_name(loss), optimizer=c, metrics=[my_iou_metric])

        model_name = self.get_model_name(stage, fold, epochs, loss, lr)
        model_checkpoint = ModelCheckpoint(model_name, monitor='val_my_iou_metric',
                                           mode='max', save_best_only=True, verbose=1)

        history = model.fit(x_train_aug, y_train_aug,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[model_checkpoint],
                            verbose=2)

        plot_history(history, 'my_iou_metric', self.get_history_path(stage, fold, epochs, loss, lr))

    def training_stage_1(self, epochs, batch_size=32, fold=0, lr=0.001, loss='binary_crossentropy'):
        # training
        stage = 1
        self.create_stage_paths(stage, fold)
        model_name = self.get_model_name(stage, fold, epochs, loss, lr)
        print(' ###################################################################################################\n',
              '####################################################################################################\n',
              'training {} stage, {} fold, {} epochs, {} batch_size\n'.format(stage, fold, epochs, batch_size),
              '{} lr, {}'.format(lr, loss))

        x_train, y_train, x_valid, y_valid = self.get_cv_data()

        # Data augmentation
        x_train_aug = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train_aug = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

        model = build_complie_model(lr=lr, loss=get_loss_by_name(loss))

        model_checkpoint = ModelCheckpoint(model_name, monitor='val_my_iou_metric',
                                           mode='max', save_best_only=True, verbose=1)

        history = model.fit(x_train_aug, y_train_aug,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[model_checkpoint],
                            verbose=2)

        plot_history(history, 'my_iou_metric', self.get_history_path(stage, fold, epochs, loss, lr))

    def training_stage_2(self, epochs, batch_size=32, fold=0, stage=2, lr=0.0005, loss='binary_crossentropy',
                         custom_objects={'my_iou_metric': my_iou_metric}, weights=''):
        # training
        self.create_stage_paths(stage, fold)
        model_name = self.get_model_name(stage, fold, epochs, loss, lr)
        print(' ###################################################################################################\n',
              '####################################################################################################\n',
              'training {} stage, {} fold, {} epochs, {} batch_size\n'.format(stage, fold, epochs, batch_size),
              '{} lr, {}'.format(lr, loss))

        x_train, y_train, x_valid, y_valid = self.get_cv_data()

        # Data augmentation
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

        if weights == '':
            weights_file = self.get_model_name(stage - 1, fold)
        else:
            weights_file = weights

        model = load_model(weights_file, custom_objects=custom_objects)

        c = optimizers.adam(lr=lr)
        model.compile(loss=get_loss_by_name(loss), optimizer=c, metrics=[my_iou_metric])

        model_checkpoint = ModelCheckpoint(model_name, monitor='val_my_iou_metric',
                                           mode='max', save_best_only=True, verbose=1)
        # reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max',
        #                               factor=0.5, patience=3, min_lr=0.0001, verbose=1)

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[model_checkpoint],
                            verbose=2)

        plot_history(history, 'my_iou_metric', self.get_history_path(stage, fold, epochs, loss, lr))

    def training_stage_3(self, epochs, batch_size=32, fold=0):
        self.training_stage_2(epochs, batch_size, fold, stage=3, lr=0.0001,
                              custom_objects={'my_iou_metric': my_iou_metric, 'lovasz_loss': lovasz_loss})

    def validate(self, stage, fold, custom_objects):
        print(' ###################################################################################################\n',
              '####################################################################################################\n',
              'validation {} stage {} fold'.format(stage, fold))

        _, _, x_valid, y_valid = self.get_cv_data()

        model = load_model(self.get_model_name(stage, fold), custom_objects=custom_objects)
        preds_valid = predict_result(model, x_valid, self.img_size_target)
        iou = get_iou_vector(y_valid, (preds_valid > 0.5))

        # Scoring for last model, choose threshold by validation data
        thresholds_ori = np.linspace(0.3, 0.8, 64)
        # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
        thresholds = np.log(thresholds_ori / (1 - thresholds_ori))

        ious = np.array(
            [iou_metric_batch(y_valid, preds_valid > threshold) for threshold in thresholds])

        # instead of using default 0 as threshold, use validation data to find the best threshold.
        threshold_best_index = np.argmax(ious)
        threshold_best = thresholds[threshold_best_index]
        iou_best = ious[threshold_best_index]
        print('Best threshold, best iou, iou with threshold 0.5:  ({}, {}, {})'.format(threshold_best, iou_best, iou))
        return threshold_best, iou_best, iou

    def predict(self, stage, fold, threshold=0.5, custom_objects={'my_iou_metric': my_iou_metric, 'lovasz_loss': lovasz_loss}):
        print(' ###################################################################################################\n',
              '####################################################################################################\n',
              'prediction {} stage, {} fold, {} threshold'.format(stage, fold, threshold))

        weights = self.get_model_name(stage=stage, fold=fold)
        model = load_model(weights, custom_objects=custom_objects)

        preds_test = predict_result(model, self.x_test, self.img_size_target)

        pred_dict = {idx: rle_encode(np.round(preds_test[i]) > threshold) for i, idx in
                     enumerate(self.test_df.index.values)}

        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv(self.get_submission_file(stage=stage, fold=fold, threshold=threshold))

    def upsample(self, img):
        if self.img_size_ori == self.img_size_target:
            return img
        return resize(img, (self. img_size_target, self. img_size_target), mode='constant', preserve_range=True)

    def downsample(self, img):
        if self.img_size_ori == self.img_size_target:
            return img
        return resize(img, (self.img_size_ori, self.img_size_ori), mode='constant', preserve_range=True)

    def show_examples(self, path):
        cv_index = 1
        train_index = self.train_all[cv_index - 1]
        evaluate_index = self.evaluate_all[cv_index - 1]

        print(train_index.shape, evaluate_index.shape)
        histall = histcoverage(self.train_df.coverage_class[train_index].values)
        print('train cv{}, number of each mask class = \n \t{}'.format(cv_index, histall))
        histall_test = histcoverage(self.train_df.coverage_class[evaluate_index].values)
        print('evaluate cv{}, number of each mask class = \n \t {}'.format(cv_index, histall_test))

        rows = 2

        fig, axes = plt.subplots(nrows=rows, ncols=8, figsize=(24, 3*rows), sharex=True, sharey=True)

        # show mask class example
        for c in range(8):
            j = 0
            for i in train_index:
                if self.train_df.coverage_class[i] == c:
                    axes[j, c].imshow(np.array(self.train_df.masks[i]))
                    axes[j, c].set_axis_off()
                    axes[j, c].set_title('class {}'.format(c))
                    j += 1
                    if (j >= rows):
                        break
        plt.savefig(path)

    def make_folds(self):
        for cv_index in range(self.cv_total):
            train_index = self.train_all[cv_index]
            evaluate_index = self.evaluate_all[cv_index]
            x_train = self.train_df.index.values[train_index].tolist()
            x_valid = self.train_df.index.values[evaluate_index].tolist()

            train_fold = pd.DataFrame()
            train_fold['id'] = x_train
            train_fold.to_csv('../folds/train_fold_{}.csv'.format(cv_index), index=False)

            valid_fold = pd.DataFrame()
            valid_fold['id'] = x_valid
            valid_fold.to_csv('../folds/valid_fold_{}.csv'.format(cv_index), index=False)

    def get_cv_data(self):
        # train_index = self.train_all[cv_index - 1]
        # evaluate_index = self.evaluate_all[cv_index - 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_train = np.array(self.train_df.images.map(self.upsample).tolist()).reshape(-1,
                                                                                    self.img_size_target,
                                                                                    self.img_size_target,
                                                                                    1)
            y_train = np.array(self.train_df.masks.map(self.upsample).tolist()).reshape(-1,
                                                                                   self.img_size_target,
                                                                                   self.img_size_target,
                                                                                   1)
            x_valid = np.array(self.valid_df.images.map(self.upsample).tolist()).reshape(-1,
                                                                                    self.img_size_target,
                                                                                    self.img_size_target,
                                                                                    1)
            y_valid = np.array(self.valid_df.masks.map(self.upsample).tolist()).reshape(-1,
                                                                                   self.img_size_target,
                                                                                   self.img_size_target, 1)
        return x_train, y_train, x_valid, y_valid

    def get_model_name(self, stage, fold, epochs, loss, lr):
        stage_ = str(stage).zfill(2)
        fold_ = str(fold).zfill(2)
        return f'../stage_{stage_}/fold_{fold_}/weights/' \
               f'weights_s-{stage_}_f-{fold_}_e-{epochs}_l-{loss}_lr-{lr}.model'

    def get_history_path(self, stage, fold, epochs, loss, lr):
        stage_ = str(stage).zfill(2)
        fold_ = str(fold).zfill(2)
        return f'../stage_{stage_}/fold_{fold_}/' \
               f'history_s-{stage_}_f-{fold_}_e-{epochs}_l-{loss}_lr-{lr}.png'

    def get_submission_file(self, stage, fold, threshold):
        stage_ = str(stage).zfill(2)
        fold_ = str(fold).zfill(2)
        thresh_ = str(round(threshold, 4))
        return self.submission_file.format(stage_, fold_, stage_, fold_, thresh_)

    def create_stage_paths(self, stage, fold):
        stage_ = str(stage).zfill(2)
        fold_ = str(fold).zfill(2)
        makedirs('../stage_{}/fold_{}/weights/'.format(stage_, fold_), exist_ok=True)
        makedirs('../stage_{}/fold_{}/submissions/'.format(stage_, fold_), exist_ok=True)
