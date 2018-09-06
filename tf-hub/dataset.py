# Utily for working with the dataset
from __future__ import division
import pandas as pd
import os
import numpy as np
from PIL import Image


labels_file = 'dataset/labels.csv'
train_data_dir = 'dataset/processed/hub/train'


labels_df = pd.read_csv(labels_file)
labels_df = labels_df.sample(frac=1).reset_index(drop=True)


g = labels_df['breed'].value_counts()
classes = list(g.keys())
n_classes = len(classes)

labels_df = labels_df.loc[labels_df['breed'].isin(classes)]

# utility functions for getting image and converting class to one hot
def get_image(filename):
    img = Image.open(filename)
    return np.asarray(img, dtype="float32") / 255.0
def get_images(filenames):
    return [get_image(os.path.join(train_data_dir, filename+'.jpg')) for filename in filenames]
def get_labels(classes_):
    return [classes.index(class_) for class_ in classes_]


# batch iteration code
batch_i_pointer = 0
n_label = len(labels_df)
n_epochs = 0
def get_batch(size=64):
    global batch_i_pointer, n_epochs
    batch_start = batch_i_pointer
    batch_end = batch_start + size
    if batch_end >= n_label:
        batch_start = batch_i_pointer = 0
        batch_end = batch_start + size
        n_epochs += 1
        
    df_ = labels_df.iloc[batch_start: batch_end]
    batch_i_pointer = batch_end
    return get_images(df_['id'].values), get_labels(df_['breed'].values)


def batch_yeild(max_epochs=2, size=64):
    global batch_i_pointer, n_epochs
    while n_epochs < max_epochs:
        batch_start = batch_i_pointer
        batch_end = batch_start + size

        if batch_end >= n_label:
            batch_start = batch_i_pointer = 0
            batch_end = batch_start + size
            n_epochs += 1

        df_ = labels_df.iloc[batch_start: batch_end]
        batch_i_pointer = batch_end

        yield get_images(df_['id'].values), get_labels(df_['breed'].values)
    