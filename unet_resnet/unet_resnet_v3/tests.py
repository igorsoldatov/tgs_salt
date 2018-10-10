import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
from skimage.transform import resize

img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

fold = 4
train_fold = f'../folds/train_fold_{fold}.csv'
valid_fold = f'../folds/valid_fold_{fold}.csv'

train_df = pd.read_csv(train_fold, index_col="id", usecols=[0])
valid_df = pd.read_csv(valid_fold, index_col="id", usecols=[0])
depths_df = pd.read_csv("../../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index) & ~depths_df.index.isin(valid_df.index)]

x_test = np.array(
            [upsample(np.array(load_img('../../input/test/images/{}.png'.format(idx), color_mode="grayscale"))) / 255 for idx in
             test_df.index]).reshape(-1, 128, 128, 1)
a = 1