from proc import preprocess
import tensorflow as tf
import h5py
import os
import multiprocessing as mp


preproc = {
    'indir': './img',
    'stride': 2,
    'patch_size': 80,  # should be multiple of 8
    'mode': 'tif',
    'shuffle': True,
    'traintest_split_rate': 0.9
}

preprocess(**preproc)
