#!/usr/bin/env python

import numpy as np
import logging
import os
from keras.applications.resnet50  import ResNet50
from keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from preprocess import preprocess_image, resize_image
from colorferet import get_dataset
from lfw import get_lfw_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Features():
    def __init__(self, dataset_path, img_width=224, img_height=224, face_crop = True, min_images_per_label=10, features_dir='../data'):
        self.img_width = img_width
        self.img_height = img_height
        self.dataset = get_lfw_dataset(dataset_path,min_images_per_label)
        self.model = ResNet50(include_top=False,weights='imagenet')
        self.batch_size = 128
        self.num_classes = len(self.dataset)
        self.face_crop = face_crop
        self.features_dir = features_dir

    def path_to_tensor(self, image_path):
        #  convert an image to tensor
        preped_img = None
        if self.face_crop == True:
            preped_img = preprocess_image(image_path, None, crop_dim=224)

        if preped_img is None:
            #  use the whole picture and first resize the image reserving its width/heigh ratio and
            ## if one side of short of 224 padding
            preped_img = resize_image(image_path, None, size=224,random_padding_border_color = True)
        logger.info("Processed image  {} shape {}".format(image_path,preped_img.shape))
        return np.expand_dims(np.array(preped_img,dtype='float'), axis=0)

    def paths_to_tensor(self,img_paths):
        # print( tqdm(img_paths))
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in img_paths]
        return np.vstack(list_of_tensors)

    def get_labels(self):
        """
        :return: ont-hot encoded labels
        """
        labels = [np.full((len(id.image_paths), 1),id.id) for id in self.dataset]
        labels = np.vstack(labels)
        file_name = os.path.join(self.features_dir,'labels_10min.npy')
        np.save(open(file_name, 'wb'),labels)
        return labels

    ## TODO: implement data generator to continously provide tensors and feed the model
    def get_feature_codes(self):
        """
        :return: feature codes array for all pictures and labels array
        """
        features = []
        logger.info("{} ids in the dataset".format(len(self.dataset)))

        def walk(data):
            # Walk through each id in a data set
            for id in data:
                    yield id

        for id in tqdm(walk(self.dataset), total = len(self.dataset),unit='id'):
            tensor_batch = self.paths_to_tensor(id.image_paths)
            inputs_batch = preprocess_input(tensor_batch)
            features_batch = self.model.predict(inputs_batch, batch_size=self.batch_size)
            # this has effect that it flattens the dimension (num_sample,1,1,2048) to (num_sample,1,2048)
            #features_batch = np.vstack(features_batch)
            features.append(features_batch)

        features = np.vstack(features)
        #logger.info("features_batch after vstack {}".format(features_batch.shape))
        # weird np.save can't do incremental save. It's lucky here that the vector output
        # is 2048 per features and total images are about 13k. If there is million images,
        # the array here is too large
        file_name = os.path.join(self.features_dir, 'bottleneck_features_face_10min.npy')
        np.save(open(file_name, 'wb'),features)
        return features

    def get_train_test_set(self,features_path, labels_path):
        """
        split the feature codes into train and test and save them in numpy array
        :param features: features array
        :param features_path:  feature array path if it is loading from file
        :return: splited train_data, train_label, test_data, test_label,test_idcs
        """
        if features_path is None or labels_path is None:
            raise ValueError("No bottleneck features available")

        if type(features_path) is str:
            features = np.load(features_path)
        elif type(features_path) is np.ndarray:
            features = features_path

        if type(labels_path) is str:
            labels = np.load(labels_path)
        elif type(labels_path) is np.ndarray:
            labels = labels_path


        ss = StratifiedShuffleSplit(n_splits=1,test_size = 0.2, random_state=0)
        train_idcs, test_idcs = next(ss.split(features, labels))
        np.save('../data/lfw_test_labels_idcs.npy',test_idcs)
        np.save('../data/lfw_test_codes.npy', features[test_idcs])
        np.save('../data/lfw_test_labels.npy', labels[test_idcs])
        np.save('../data/lfw_train_codes.npy', features[train_idcs])
        np.save('../data/lfw_train_labels.npy', labels[train_idcs])
        return features[train_idcs],labels[train_idcs],features[test_idcs],labels[test_idcs],test_idcs


# generate bottleneck features offline
if __name__ == '__main__':
    feature = Features('/Volumes/ML/lfw/', 224, 224, face_crop=True, min_images_per_label=10,
                       features_dir='../data/')
    labels =feature.get_labels()
    print(labels.shape)
    codes = feature.get_feature_codes()
    print(codes.shape)
    codes = np.load('../data/bottleneck_features_face_10min.npy')
    labels = np.load('../data/labels_10min.npy')
    train_data,train_labels,test_data,test_labels, test_idcs= feature.get_train_test_set(codes,labels)
    print(train_data.shape,train_labels.shape,test_data.shape,test_labels.shape)
    print(test_idcs)



