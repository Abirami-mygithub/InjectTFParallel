"""
Filename:           gtsrb_dataset.py 
File Description:   GTSRB (Geraman Traffic Sign Recognition Benchmark) dataset must be dowloaded from https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
                    before executing gtsrb_dataset.py. Place the dataset in current directory with a new folder named gtsrb
                    This file performs visualization of dataset, data preprocessing and splits the dataset into training and test dataset.
                    There are getter functions for training and test dataset.
Created by:         Abirami Ravi - University of Stuttgart (abirami1429@gmail.com)
References:         https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
                    https://benchmark.ini.rub.de/gtsrb_news.html
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

import constants

class GTSRB_DATASET:

    def __init__(self):

        """Note: Kindly modify the path containing the dataset. 
        """
        data_dir =      "G:\InjectTFParallel\datasets\gtsrb"
        train_path =    "G:\InjectTFParallel\datasets\gtsrb\Train"
        test_path =     "G:\InjectTFParallel\datasets\gtsrb\Test"

        # Uncomment following two lines to check the contents in training and test dataset
        #print(os.listdir(train_path))
        #print(os.listdir(test_path))

        nb_classes_gtsrb = len(os.listdir(train_path))

        # Resizing the images to 32x32x3
        IMG_HEIGHT = constants.GTSRB_IMG_HEIGHT
        IMG_WIDTH = constants.GTSRB_IMG_WIDTH
        channels = constants.GTSRB_channels

        #Defining the class labels
        classes = constants.GTSRB_LABELS
        
        #Combining the images with their corresponding class labels
        folders = os.listdir(train_path)
        train_number = []
        class_num = []
        
        for folder in folders:
            train_files = os.listdir(train_path + '/' + folder)
            train_number.append(len(train_files))
            class_num.append(classes[int(folder)])
        print("Length of the training dataset list:", len(train_number))
        print("Total no of classes:", class_num)
        print(train_number)
        zipped_lists = zip(train_number, class_num)
        sorted_pairs = sorted(zipped_lists)
        tuples = zip(*sorted_pairs)
        train_number, class_num = [ list(tuple) for tuple in  tuples]
        print("example of tuple created:", train_number[20], class_num[20])

        #Collecting the training dataset       
        image_data = []
        image_labels = []
        image_filenames = []
        
        for i in range(nb_classes_gtsrb):
            path = data_dir + '/Train/' + str(i)
            images = os.listdir(path)
            print(path)
            for img in images:
                try:
                    image = cv2.imread(path + '/' + img)
                    image_fromarray = Image.fromarray(image, 'RGB')
                    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
                    image_data.append(np.array(resize_image))
                    image_labels.append(i)
                except:
                    print("Error in " + img)
        
        # Changing the list to numpy array
        image_data = np.array(image_data)
        image_labels = np.array(image_labels)
        
        print(image_data.shape, image_labels.shape)

        #Shuffling the training data
        shuffle_indexes = np.arange(image_data.shape[0])
        np.random.shuffle(shuffle_indexes)
        image_data = image_data[shuffle_indexes]
        image_labels = image_labels[shuffle_indexes]

        #Splitting the data into train and validation set
        X_train, X_test, y_train, y_test = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)
        
        X_train = X_train/255
        X_test = X_test/255

        #One hot encoding the labels
        y_train = tf.keras.utils.to_categorical(y_train, no_of_classes)
        y_test = tf.keras.utils.to_categorical(y_val, no_of_classes)
        
        print(y_train.shape)
        print(y_test.shape)


