#!/usr/bin/env python

import numpy as np
import logging
import os
from keras.applications.resnet50  import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

from preprocess import preprocess_image, resize_image
from lfw import get_lfw_dataset,get_random_lfw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Features():
    def __init__(self, dataset_path, img_width=224, img_height=224, face_crop = True,
                 min_images_per_label=10, max_images_per_label = 1000, mode='softmax',features_dir='../data'):
        self.img_width = img_width
        self.img_height = img_height
        self.dataset = get_lfw_dataset(dataset_path,min_images_per_label,max_images_per_label)
        self.random_draw_test = get_random_lfw(dataset_path,200, min_images_per_label/2)
        if mode == 'softmax':
            self.model = ResNet50(include_top=False, weights='imagenet')
        else:
            resnet_model = ResNet50(include_top=False, weights='imagenet')
            self.model = Model(inputs=resnet_model.input,outputs=resnet_model.get_layer('activation_49').output)
        self.model.summary()
        self.batch_size = 128
        self.face_crop = face_crop
        self.features_dir = features_dir

    def path_to_tensor(self, image_path):
        #  convert an image to tensor
        preped_img = None
        if self.face_crop == True:
            preped_img = preprocess_image(image_path, None, crop_dim=224)
            if preped_img is None:
                return []
        else:
            #  use the whole picture and first resize the image reserving its width/heigh ratio and
            ## if one side of short of 224 padding
            preped_img = resize_image(image_path, None, size=224,random_padding_border_color = True)
        logger.info("Processed image  {} shape {}".format(image_path,preped_img.shape))
        return np.expand_dims(np.array(preped_img,dtype='float'), axis=0)

    def paths_to_tensor(self,img_paths):
        # print( tqdm(img_paths))
        list_of_tensors = []
        for img_path in img_paths:
            tensor = self.path_to_tensor(img_path)
            if len(tensor ) > 0:
                list_of_tensors.append(tensor)
        if len(list_of_tensors) >0:
            list_of_tensors =  np.vstack(list_of_tensors)
            return list_of_tensors
        else:
            return []

    def get_codes(self,dataset,feature_file_name, label_file_name):
        """
        :return: feature codes array for all pictures and labels array
        """
        features = []
        logger.info("{} ids in the dataset".format(len(dataset)))
        labels = []
        def walk(data):
            # Walk through each id in a data set
            for id in data:
                    yield id

        for id in tqdm(walk(dataset), total = len(dataset),unit='id'):
            tensor_batch = self.paths_to_tensor(id.image_paths)
            if len(tensor_batch)==0:
                continue
            logger.info("tensor  shape {}".format(tensor_batch.shape))
            inputs_batch = preprocess_input(tensor_batch)
            features_batch = self.model.predict(inputs_batch)#
            # this has effect that it flattens the dimension (num_sample,1,1,2048) to (num_sample,1,2048)
            #features_batch = np.vstack(features_batch)
            label_batch = np.full((len(inputs_batch), 1), id.id)
            logger.info("features_batch  shape {}".format(features_batch.shape))
            logger.info("label_batch  shape {}".format(label_batch.shape))
            features.append(features_batch)
            labels.append(label_batch)

        features = np.vstack(features)
        labels = np.vstack(labels)
        #logger.info("features_batch after vstack {}".format(features_batch.shape))
        # weird np.save can't do incremental save. It's lucky here that the vector output
        # is 2048 per features and total images are about 5k. If there is million images,
        # the array here is too large
        file_name = os.path.join(self.features_dir, feature_file_name)
        np.save(open(file_name, 'wb'),features)

        file_name = os.path.join(self.features_dir,label_file_name)
        np.save(open(file_name, 'wb'),labels)


        return features,labels

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

        file_name = os.path.join(self.features_dir, 'test_labels_idcs.npy')
        np.save(file_name,test_idcs)

        file_name = os.path.join(self.features_dir, 'test_codes.npy')
        np.save(file_name,features[test_idcs])

        file_name = os.path.join(self.features_dir, 'test_labels.npy')
        np.save(file_name,labels[test_idcs])

        file_name = os.path.join(self.features_dir, 'train_codes.npy')
        np.save(file_name,features[train_idcs])

        file_name = os.path.join(self.features_dir, 'train_labels.npy')
        np.save(file_name,labels[train_idcs])

        return features[train_idcs],labels[train_idcs],features[test_idcs],labels[test_idcs],test_idcs


# generate bottleneck features offline
if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--min', type=int, action='store', default=10, dest='min_images_per_label',
                        help='min images per label')
    parser.add_argument('--max', type=int, action='store', default=1000, dest='max_images_per_label',
                        help='max images per label')
    parser.add_argument('--mode', type=str, action='store', default='softmax', dest='mode',
                        help='generate bottleneck for softmax classifier or triplet loss classifier')

    args = parser.parse_args()

    if os.path.exists(args.input_dir) == False:
        logger.error("input {} doesn't exist!".format(args.input_dir))
    if os.path.exists(args.output_dir) == False:
        os.makedirs(args.output_dir)

    feature = Features(args.input_dir, 224, 224, face_crop=True, min_images_per_label=args.min_images_per_label,
                       max_images_per_label = args.max_images_per_label, mode = args.mode,
                        features_dir=args.output_dir)
    codes, labels = feature.get_codes(feature.dataset, feature_file_name ='bottleneck_features.npy',label_file_name='labels.npy')
    print(labels.shape)
    print(codes.shape)
    train_data,train_labels,test_data,test_labels, test_idcs= feature.get_train_test_set(codes,labels)
    print(train_data.shape,train_labels.shape,test_data.shape,test_labels.shape)

    codes, labels = feature.get_codes(feature.random_draw_test, feature_file_name ='random_draw_codes.npy',label_file_name='random_draw_labels.npy')
    print(labels.shape)
    print(codes.shape)
