#!/usr/bin/env python
import os
import logging
import numpy as np
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ImageClass():
    def __init__(self, name,id, image_paths):
        self.name = name
        self.image_paths = image_paths
        self.id = id

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def dump_dataset(file_name, dataset):
    """
    :param file_name:
    :param dataset:  dataset to dump
    :return:
    """
    wr = open (file_name, 'w')
    wr.write('id' + ', '+'label' + ', ' + 'name' + ', ' + 'path' + '\n')
    i = 0
    for item in dataset:
        for images in item.image_paths:
            wr.write(str(i) + ', '+ str(item.id) + ', ' + item.name + ', ' +images + '\n' )
            i +=1
    wr.close()


def get_lfw_dataset(input_directory,min_images_per_label=10):
    dataset = []
    ids = os.listdir(input_directory)
    ids.sort()
    num_classes = len(ids)

    if min_images_per_label < 2:
        raise ValueError("min_images_per_label has to be at least 2 otherwise can't split training and testing data")

    for i in range(num_classes):
        id_name = ids[i]
        facedir = os.path.join(input_directory, id_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            if len(images) >= min_images_per_label:
                image_paths = [os.path.join(facedir, img) for img in images]
                dataset.append(ImageClass(id_name,i, image_paths))
            else:
                logger.info("Not enough pictures to train for ID {} Skipping..".format(id_name))


    ##debug
    file_name = 'lfw_dataset.csv'
    dump_dataset(file_name, dataset)

    return dataset

## split data for training and testing
def split_dataset(dataset, test_size = 0.2):
    """

    :param test_size: split ratio for test and train
    :return:  train_set and test_set
    """
    train_set = []
    test_set = []
    train_dir = 'train'
    test_dir = 'test'

    ## in a given id folder at least there needs to be 2 images
    for id in dataset:
        paths = id.image_paths
        np.random.shuffle(paths)
        split = int(round(test_size * len(paths)))
        if not os.path.exists(train_dir + '/' + id.name):
            os.makedirs(train_dir + '/' + id.name)
        if not os.path.exists(test_dir + '/' + id.name):
            os.makedirs(test_dir + '/' + id.name)
        test_set.append(ImageClass(id.name, id.id, paths[0:split]))
        train_set.append(ImageClass(id.name, id.id, paths[split:-1]))

        for filename in paths[0:split]:
            shutil.copy(filename, test_dir + '/' + id.name)
        for filename in paths[split:-1]:
            shutil.copy(filename, train_dir + '/' + id.name)

    ##debug
    file_name = 'train_set.csv'
    dump_dataset(file_name, train_set)

    file_name = 'test_set.csv'
    dump_dataset(file_name, test_set)

    return train_set, test_set


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dataset = get_lfw_dataset('/Volumes/ML/lfw/',min_images_per_label=10)
    #train_set, test_set = split_dataset(dataset, 0.2)
