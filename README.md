# Schools Classification from Satellite Imagery

A model that identifies whether a school is present in a satellite image. This is part of a project for mapping schools all over the world!

# Training the model

Assuming a path to two .hdf5 files that contain the dataset run: 

`python main.py --targetFolder <.hdf5 folder path>`

The hdf5 files are a specific split of the dataset (90% training 10% testing), from more than 4000 images that are evenly split between the "school present" and "school not present" categories.

# Dataset

Our dataset contains satellite images with area coverage of 400m by 400m. Each image has a label indicating whether a school is present or not within the image. To make our model more robust, schools are not always at the center of the image. We also balanced the training data by collecting of images without schools from places near the schools.

For example, images that include schools include:

<img src=https://github.com/taqadam/SchoolClassification/blob/master/DatasetExamples/Ex1School.jpg width=224px height=224px><img src=https://github.com/taqadam/SchoolClassification/blob/master/DatasetExamples/Ex2School.jpg width=224px height=224px>

whereas images without schools include:

<img src=https://github.com/taqadam/SchoolClassification/blob/master/DatasetExamples/Ex1NoSchool.jpg width=224px height=224px><img src=https://github.com/taqadam/SchoolClassification/blob/master/DatasetExamples/Ex2NoSchool.jpg width=224px height=224px>

# Pre-processing

We normalize the images based on the mean and standard deviation of a subset of the training dataset. For data augmentation we use random flipping (vertically and horizontally), as satellite images are typically invariant to orientation changes. We downscale the images to 224x224 or 518x518 crops.

# Model

We have trained both a mobilenetV2 model and a custom-made model that is architecturally similar to VGG16.
Our model achieves approximately 80% accuracy at 20 epochs of training.

# Structure

`datasplitter.py` - is used to create the two .hdf5 files from two folders of training data (one for each class)

`cycler.py` - an implementation of cyclical learning rate to help train the model

`generator.py` - data augmentation and processing before it is given to the network

`mymodels.py` - contain the model architectures and optimizers

`main.py` - the entry point for training the model

`tensorboardUtils.py` - for visualizing results, producing accuracy and loss figures and calculating the confusion matrices
