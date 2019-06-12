# The goal of this script is to crop images of schools so that there are
# two types of image sources. One with a school at its center, and one with
# a school not present or present in a corner.

import sys
import argparse
import cv2
import glob
import numpy as np
import re
import h5py
import math
import random

class DatasetCropper():
    def __init__(self, folderSourceNegative, folderSourcePositive, folderTarget, imageSize=(256,256), cropSize=(224,224), testSplit=0.1):
        self.folderSourceNegative = folderSourceNegative
        self.folderSourcePositive = folderSourcePositive
        self.folderTarget = folderTarget
        self.mean = None # list of mean values for each channel in BGR format or None to re-calculate
        self.std = None
        self.imageSize = imageSize
        self.cropSize = cropSize
        self.testSplit = testSplit
        self.createDatafiles()


    def createDatafiles(self):
        self.hdf5FileTrain = h5py.File(self.folderTarget+"/datasetTrain.hdf5","w")
        self.hdf5FileTest = h5py.File(self.folderTarget+"/datasetTest.hdf5","w")
        self.imageDataFile = {"Train":[0,None],"Test":[0,None]}
        self.labelDataFile = {"Train":[None], "Test":[None]}

    def addImageLabelData(self, image, info, mode="Train"):
        index = self.imageDataFile[mode][0]
        dset = self.imageDataFile[mode][1]
        dsetLabel = self.labelDataFile[mode]
        dset[index,:,:,:] = image
        dsetLabel[index,0] = int(info["label"])
        dsetLabel[index,1] = float(info["lon"])
        dsetLabel[index,2] = float(info["lat"])
        self.imageDataFile[mode][0] += 1

    def validImage(self, img):
        #
        # Some images fail to be loaded and are empty
        #

        imgMean = np.mean(img)

        # Through visual inspection, 224 is the minimum average value of pixels
        # in all images with no content

        # Images with clouds or very high brightness are contained within the 210-224 value range
        # We curate the dataset by removing them

        if imgMean > 210:
            return False

        return True

    def crop(self, image):
        cropx, cropy = self.cropSize
        y,x,c = image.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        return image[starty:starty+cropy,startx:startx+cropx]

    def preprocess(self, image):
        image = self.crop(image)
        image = image.astype('float32')
        image -= np.array(self.mean,dtype='float32')
        image /= np.array(self.std,dtype='float32')

        return image

    def processImage(self, image, imagePath, hdf5File, label=0, mode="Train"):
        fileName = imagePath.split("/")[-1]
        gpsLocationMixed = fileName[:-4] # removing file extension
        if label == 0:
            gpsLocation = re.search(r"(-?[0-9\.]+)_negEx-{1}(-?[0-9\.]+)_negEx", gpsLocationMixed) # splitting the two coordinates
        else:
            gpsLocation = re.search(r"(-?[0-9\.]+)-{1}(-?[0-9\.]+)", gpsLocationMixed) # splitting the two coordinates

        lat, lon = gpsLocation.group(1), gpsLocation.group(2)

        finalImage = self.preprocess(image)
        info = {"label":label,"lon":lon,"lat":lat}
        self.addImageLabelData(finalImage, info, mode)

    def calculateMeanStd(self, dataset):
        # Sample mean only from valid training images (not just positive)
        # do this at a later step for both this and std

        sampleEstimate = min(500,len(dataset))
        if self.mean is None:
            self.mean = [0,0,0]
            for i, imagePath in enumerate(dataset):
                if i % 50 == 0:
                    print("mean at {}/{}".format(i, sampleEstimate))
                img = cv2.imread(imagePath)
                img = cv2.resize(img,(self.imageSize[0],self.imageSize[1]))
                # mean is in BGR mode

                self.mean  += np.mean(img,axis=(0,1))
                if sampleEstimate <= i:
                    break

            self.mean = [self.mean[i]/sampleEstimate for i in range(3)]
            print("Mean estimated at {}".format(self.mean))

        if self.std is None:
            var = [0,0,0]
            for i, imagePath in enumerate(dataset):
                if i % 50 == 0:
                    print("mean at {}/{}".format(i, sampleEstimate))
                img = cv2.imread(imagePath)
                img = cv2.resize(img,(self.imageSize[0],self.imageSize[1]))
                img = img.astype('float32')
                img -= np.array(self.mean,dtype='float32')
                # mean is in BGR mode
                imgS = np.square(img)
                var += np.sum(imgS, axis=(0,1))/(self.imageSize[0]*self.imageSize[1])
                if sampleEstimate <= i:
                    break

            var = [var[i]/sampleEstimate for i in range(3)]
            self.std = [math.sqrt(var[i]) for i in range(3)]
            print("Std estimated at {}".format(self.std))

    def processFolder(self):

        sourceImagesN = glob.glob(self.folderSourceNegative+"*.jpg")
        random.shuffle(sourceImagesN)
        sourceImagesP = glob.glob(self.folderSourcePositive+"*.jpg")
        random.shuffle(sourceImagesP)

        if len(sourceImagesP) == 0:
            raise Exception('Source image folder contains no .jpg images')
        if len(sourceImagesN) == 0:
            raise Exception('Source image folder contains no .jpg images')

        sourceImageNValid = []
        sourceImagePValid = []

        print("Initial lengths for positive and negative datasets are P:{} N:{}".format(len(sourceImagesP),len(sourceImagesN)))

        for i, imagePath in enumerate(sourceImagesN):
            img = cv2.imread(imagePath)
            img = cv2.resize(img,(224,224))
            if self.validImage(img):
                sourceImageNValid.append(imagePath)

        print("Removed {} invalid negative images".format(len(sourceImagesN)-len(sourceImageNValid)))

        for i, imagePath in enumerate(sourceImagesP):
            img = cv2.imread(imagePath)
            img = cv2.resize(img,(224,224))
            if self.validImage(img):
                sourceImagePValid.append(imagePath)

        print("Removed {} invalid positive images".format(len(sourceImagesP)-len(sourceImagePValid)))

        print("Current lengths for positive and negative datasets become P:{} N:{}".format(len(sourceImagePValid),len(sourceImageNValid)))
        minAmountOfImages = (min(len(sourceImagePValid), len(sourceImageNValid)))
        sourceImagePValid = sourceImagePValid[:minAmountOfImages]
        sourceImageNValid = sourceImageNValid[:minAmountOfImages]

        datasetSplit = {"Train":{"P":[],"N":[]},"Test":{"P":[],"N":[]}}

        for i, imagePath in enumerate(sourceImageNValid):
            if i*1.0/minAmountOfImages > 1-self.testSplit:
                datasetSplit["Test"]["N"].append(imagePath)
            else:
                datasetSplit["Train"]["N"].append(imagePath)

        for i, imagePath in enumerate(sourceImagePValid):
            if i*1.0/minAmountOfImages > 1-self.testSplit:
                datasetSplit["Test"]["P"].append(imagePath)
            else:
                datasetSplit["Train"]["P"].append(imagePath)

        print("Train Positive {} Train Negative {} Test Positive {} Test Negative {}".format(len(datasetSplit["Train"]["P"]),
                                                                                    len(datasetSplit["Train"]["N"]),
                                                                                    len(datasetSplit["Test"]["P"]),
                                                                                    len(datasetSplit["Test"]["N"])))
        # calculate mean and standard deviation per channel on valid images
        self.calculateMeanStd(datasetSplit["Train"]["P"]+datasetSplit["Train"]["N"])
        trainMeta = self.hdf5FileTrain.create_dataset('meta',(6,),'float32')
        testMeta = self.hdf5FileTest.create_dataset('meta',(6,),'float32')
        trainMeta[:] = np.array(self.mean+self.std)
        testMeta[:] = np.array(self.mean+self.std)

        self.imageDataFile["Train"][1] = self.hdf5FileTrain.create_dataset('images',(len(datasetSplit["Train"]["P"])+len(datasetSplit["Train"]["N"]),self.cropSize[0], self.cropSize[1],3),'float32')
        self.imageDataFile["Test"][1] = self.hdf5FileTest.create_dataset('images',(len(datasetSplit["Test"]["P"])+len(datasetSplit["Test"]["N"]),self.cropSize[0], self.cropSize[1],3),'float32')

        self.labelDataFile["Train"] = self.hdf5FileTrain.create_dataset('labels',(len(datasetSplit["Train"]["P"])+len(datasetSplit["Train"]["N"]),3),'float32')
        self.labelDataFile["Test"] = self.hdf5FileTest.create_dataset('labels',(len(datasetSplit["Test"]["P"])+len(datasetSplit["Test"]["N"]),3),'float32')

        for keyMode in datasetSplit.keys():
            for keyClass in datasetSplit[keyMode].keys():
                for i, imagePath in enumerate(datasetSplit[keyMode][keyClass]):
                    if i % 50 == 0:
                        print("Processing {} / {} from {},{}".format(i, len(datasetSplit[keyMode][keyClass]),keyMode,keyClass))
                    label = 1
                    if keyClass == "N":
                        label = 0
                    datafile = self.hdf5FileTrain
                    if keyMode == "Test":
                        datafile = self.hdf5FileTest
                    img = cv2.imread(imagePath)
                    img = cv2.resize(img,(self.imageSize[0], self.imageSize[1]))
                    self.processImage(img, imagePath, datafile, label, keyMode)

parser = argparse.ArgumentParser(description="Acquisition of folder paths")
parser.add_argument('--folderSourcePositive', default="./Datasets/DatasetSchools/")
parser.add_argument('--folderSourceNegative', default="./Datasets/DatasetNoSchools/")
parser.add_argument('--folderTarget', default="/<path to datasets>/Datasets/")
args = parser.parse_args()

datasetCropper = None
if "224" in args['folderTarget']:
    datasetCropper = DatasetCropper(args.folderSourceNegative, args.folderSourcePositive, args.folderTarget, imageSize=(256,256), cropSize=(224,224), testSplit=0.1)

if "518" in args['folderTarget']:
    datasetCropper = DatasetCropper(args.folderSourceNegative, args.folderSourcePositive, args.folderTarget, imageSize=(576,576), cropSize=(518,518), testSplit=0.1)

if "1152" in args['folderTarget']:
    datasetCropper = DatasetCropper(args.folderSourceNegative, args.folderSourcePositive, args.folderTarget, imageSize=(1280,1280), cropSize=(1152,1152), testSplit=0.1)

if datasetCropper is not None:
    datasetCropper.processFolder()
