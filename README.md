# SchoolsClassification

We develop a model that can identify whether a satellite image is of a school or not. This is part of a project for mapping schools all over the world!

# Training the model

Assuming a path to two .hdf5 files that contain the dataset, we run the program with "python main.py --targetFolder <.hdf5 folder path>".
The hdf5 files are a specific split of the dataset (90% training 10% testing), from more than 4000 images that are evenly split between the "school present" and "school not present" categories.

# Dataset

Our dataset includes images of schools and areas without schools. The typical area covered by the images is 400m by 400m. The schools are not always at the center of the picture. In order to make the problem more challenging, the collection of images without schools were taken from places near the schools, so as to keep the distribution of features similar between pairs of opposite-class images.

# Pre-processing

We normalize the images based on the mean and standard deviation of a subset of the training dataset. For data augmentation we use random flipping (vertically and horizontally), as satellite images are typically invariant to orientation changes. We downscale the images to 224x224 or 518x518 crops.

# Model

We have trained both a mobilenetV2 model and a custom-made model that is architecturally similar to VGG16.
Our model achieves approximately 80% accuracy at 20 epochs of training.

# Structure

datasplitter.py - is used to create the two .hdf5 files from two folders of training data (one for each class)

cycler.py - an implementation of cyclical learning rate to help train the model

generator.py - data augmentation and processing before it is given to the network

mymodels.py - contain the model architectures and optimizers

main.py - the entry point for training the model

tensorboardUtils.py - for visualizing results, producing accuracy and loss figures and calculating the confusion matrices
