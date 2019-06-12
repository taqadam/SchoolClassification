import sys
import argparse
import cv2
import glob
import numpy as np
import re
import h5py
import skimage

from generator import DataGenerator
import tensorflow as tf
from keras.optimizers import Adam
import keras
from mymodels import ModelBundle224, ModelBundle518, ModelBundleMobilenetV2
from tensorboardUtils import customTensorboardInfo
from trainer import Trainer
import os
import tensorflow as tf
from cycler import CyclicLR

# Disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

imageScale = 224 # 224, 518
parser = argparse.ArgumentParser(description="Acquisition of folder paths")
parser.add_argument('--folderTarget', default="/<path to Dataset>/Datasets/{}/".format(imageScale))
args = parser.parse_args()

# Create the dataset generators and the model/optimizer bundle

dGenTrain = DataGenerator(args.folderTarget, mode="Train",imageSize=(imageScale,imageScale), batch_size=16)
dGenTest = DataGenerator(args.folderTarget, mode="Test",imageSize=(imageScale,imageScale), batch_size=16)
modelBundle = ModelBundle224(lr=0.0001,imageSize=(imageScale, imageScale, 3))

# Default and custom callbacks for tensorboard visualizations
logdir = "./logdir/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
clr = CyclicLR()
tbi_callback = customTensorboardInfo(imageScale, 'Visualizations',logdir,dGenTest)
# Determine hyperparameters
modelBundle.modelOptimizer()

trainer = Trainer(dGenTrain, dGenTest, modelBundle, tensorboardCallbacks=True)
trainer.train(epochs=300,callbacks=[tensorboard_callback, tbi_callback, clr])
