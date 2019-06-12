import numpy as np
import keras
import h5py
import random
import cv2

class DataGenerator(keras.utils.Sequence):
    def __init__(self, folderTarget, imageSize = (224,224),batch_size=32, mode="Train"):

        self.dataListLength = -1
        self.mode = mode

        if mode == "Train":
            self.hdf5File = h5py.File(folderTarget+'/datasetTrain.hdf5', 'r')
        else:
            self.hdf5File = h5py.File(folderTarget+'/datasetTest.hdf5', 'r')

        self.dataListLength = self.hdf5File['images'].shape[0]
        self.mean = self.hdf5File['meta'][0:3]
        self.std = self.hdf5File['meta'][3:6]

        self.n_classes = 2
        self.dim = [imageSize[0], imageSize[1], 3]
        self.batch_size = batch_size
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.dataListLength / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.dataListLength)
        random.shuffle(self.indexes)

    def augmentImage(self, img):
        if self.mode == "Test":
            return img

        if random.random() < 0.5:
            img = np.flip(img,1)
        if random.random() < 0.5:
            img = np.flip(img,0)

        return img

    def __data_generation(self, list_IDs_temp):

        # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim[0:2], self.dim[2]))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.augmentImage(self.hdf5File['images'][ID][:,:,:])

            # Store class
            y[i] = self.hdf5File['labels'][ID][0]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
