#!/usr/bin/env python
import os
import logging
import numpy as np
import shutil
import matplotlib.pyplot as plt

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



def get_lfw_dataset(input_directory,min_images_per_label=10,max_images_per_label = 1000):
    dataset = []
    ids = os.listdir(input_directory)
    ids.sort()
    num_classes = len(ids)
    stats = np.zeros(num_classes,dtype='int')

    # if min_images_per_label < 2:
    #     raise ValueError("min_images_per_label has to be at least 2 otherwise can't split training and testing data")

    for i in range(num_classes):
        id_name = ids[i]
        facedir = os.path.join(input_directory, id_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            if len(images) >= min_images_per_label:
                stats[i] = len(images)
                if len(images)< max_images_per_label :
                    image_paths = [os.path.join(facedir, img) for img in images]
                    dataset.append(ImageClass(id_name,i, image_paths))
                else:
                    choices  = np.random.choice(images, max_images_per_label)
                    image_paths = [os.path.join(facedir, img) for img in choices]
                    stats[i] = len(image_paths)
                    dataset.append(ImageClass(id_name,i, image_paths))
            else:
                logger.info("Not enough pictures or too many pictures(skewed) to train for ID {} Skipping..".format(id_name))

    ##debug
    file_name = 'lfw_dataset.csv'
    dump_dataset(file_name, dataset)

    stats = stats[stats > 0]
    bin_width = 5
    n, bins, patches = plt.hist(stats, bins=range(0, max(stats) + bin_width, bin_width),normed=False, facecolor='g') # ss, alpha=0.75)


    plt.xlabel('number of picture per identity')
    plt.ylabel('number of identity density')
    plt.title('Histogram of  number of picture')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    #plt.xscale('log')
    #plt.ylim((0.0,1.0))
    plt.grid(True)
    plt.show()

    return dataset

def draw_histogram(dataset):
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

def get_random_lfw(input_directory,num_pictures,min_images_per_label=20):
    dataset = []
    ids = os.listdir(input_directory)
    ids.sort()
    num_classes = len(ids)

    for i in range(num_classes):
        id_name = ids[i]
        facedir = os.path.join(input_directory, id_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            if len(images) < min_images_per_label:
                image_paths = [os.path.join(facedir, img) for img in images]
                image_path = np.random.choice(image_paths)
                dataset.append(ImageClass(id_name,i, [image_path]))

    random_draw_set = np.random.choice(dataset, num_pictures)

    file_name = 'lfw_random_draw.csv'
    dump_dataset(file_name, random_draw_set)
    return random_draw_set

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
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--min', type=int, action='store', default=10, dest='min_images_per_label',
                        help='min images per label')
    parser.add_argument('--max', type=int, action='store', default=1000, dest='max_images_per_label',
                        help='max images per label')
    parser.add_argument('--random_draw', type=int, action='store', default=100, dest='random_draw_num',
                        help='number of random draw pictures')

    args = parser.parse_args()

    if os.path.exists(args.input_dir) == False:
        logger.error("input {} doesn't exist!".format(args.input_dir))

    dataset = get_lfw_dataset(args.input_dir ,min_images_per_label=args.min_images_per_label
                              , max_images_per_label = args.max_images_per_label)
    random_draw = get_random_lfw(args.input_dir, num_pictures=args.random_draw_num)

