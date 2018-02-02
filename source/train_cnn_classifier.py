#!/usr/bin/env python

import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import  EarlyStopping
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from evaluate_classifier import accuracy, multiclass_roc_plot,false_accept, false_reject,equal_error_rate,det_curve_plot
from bottleneck_features import Features
from sklearn.metrics import precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CNNClassifier():
    def __init__(self,dataset_path, classfier_filename,num_class,min_images_per_label=5,use_triplet_loss=False ):
        self.feature = Features(dataset_path, 224, 224, num_class,
                                face_crop=False, min_images_per_label = min_images_per_label, features_dir='../data/')

        #self.num_class = num_class  # almost not important
        self.use_triplet_loss = use_triplet_loss
        self.classifier_filename = classfier_filename
        self.num_embeddings = 128
        self.lb = LabelBinarizer()
        K.set_image_data_format('channels_last')

    def prepare(self):
        labels = np.load('../data/labels_5min.npy')
        self.train_codes = np.load('../data/train_codes.npy')
        self.test_codes = np.load('../data/test_codes.npy')
        self.train_labels = np.squeeze(np.load('../data/train_labels.npy'))
        self.test_labels = np.squeeze(np.load('../data/test_labels.npy'))

        ## label to binarized codes
        self.lb.fit(np.squeeze(labels))
        self.trm_test_labels = self.lb.transform(self.test_labels)
        self.trm_train_labels = self.lb.transform(self.train_labels)

        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_idcs, val_idcs = next(ss.split(self.train_codes, self.trm_train_labels))
        self.x_codes = self.train_codes[train_idcs]
        self.y_labels = self.trm_train_labels[train_idcs]
        self.val_codes = self.train_codes[val_idcs]
        self.val_labels = self.trm_train_labels[val_idcs]

        logger.info("train codes shape {}".format(self.x_codes.shape))
        logger.info("validation codes shape {}".format(self.val_codes.shape))
        logger.info("train label shape {}".format(self.y_labels.shape))
        logger.info("validation label shape {}".format(self.val_labels.shape))

        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D(name='global_average', input_shape=self.x_codes.shape[1:]))
        if not self.use_triplet_loss:
           # classifier
            self.model.add(Dense(len(np.unique(self.test_labels)), activation='softmax', name='prediction'))
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            self.model.summary()
        else:
            # TODO implement triplet lose
            self.model.add(Dense(self.num_embeddings, activation='LeakyReLU', name='prediction'))


    def train(self):

        ## split training set further for training and validation set

        start_time = time.time()
        logger.info('Training start...' )
        checkpointer = ModelCheckpoint(filepath=self.classifier_filename,
                                       verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=20)
        evaluation = self.model.fit(self.x_codes, self.y_labels,
                        validation_data=(self.val_codes, self.val_labels),
                        epochs=1000, batch_size=32, callbacks=[checkpointer,early_stopping], verbose=1)
        logger.info('Completed in {} seconds'.format(time.time() - start_time))
        logger.info("Training result {}".format(evaluation))
        pickle.dump(evaluation.history, open("../model/cnn_train_history.pickle", "wb"))
        return evaluation

    def evaluation(self):
        # summarize history for accuracy
        with open("../model/cnn_train_history.pickle", 'rb') as pickle_file:
            history = pickle.load(pickle_file)
        plt.figure(figsize=(5,3))
        plt.subplot(1, 2, 1)
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        #plt.show()
        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        self.model.load_weights(self.classifier_filename)
        predict_probs = self.model.predict(self.test_codes)
        logger.info("prediction prability matrix shape {}".format(predict_probs.shape) )
        ## change probability to binarized label
        predictions = (predict_probs == predict_probs.max(axis=1)[:, None]).astype(int)
        logger.info("prediction  matrix shape {}".format(predictions.shape))
        np.save('../model/cnn_prediction_prob.npy', predict_probs)

        predictions = self.lb.inverse_transform(predictions)
        test_idcs = np.load('../data/test_labels_idcs.npy')
        # what went wrong
        for test_id in range(len(self.test_labels)):
            if predictions[test_id] != self.test_labels[test_id]:
                ## label number starts from 1 in the colorferet
                logger.info(
                    "data id  {} predicted label {}, true label {}".format(test_idcs[test_id], predictions[test_id] + 1,
                                                                           self.test_labels[test_id] + 1))
        logger.info(
            "weighted precision score {:.4f}".format(precision_score(self.test_labels, predictions, average='weighted')))
        logger.info("weighted recall score {:.4f}".format(recall_score(self.test_labels, predictions, average='weighted')))
        print("accuracy: {0:.4f}".format(accuracy(predict_probs, self.test_labels)))
        print("false_accept: {0:.4f}".format(false_accept(predict_probs, self.test_labels)))
        print("false_reject: {0:.4f}".format(false_reject(predict_probs, self.test_labels)))
        det_curve_plot(predict_probs, self.test_labels)
        errate, thres = equal_error_rate(predict_probs, self.test_labels)
        print("equal error rate: ", errate)
        print("threshold: ", thres)
        #multiclass_roc_plot(predictions, self.test_labels, len(np.unique(self.test_labels)))


if __name__=='__main__':
    clr = CNNClassifier('/Volumes/ML/ColorFeret_Test/','../model/weights_soft_max.hdf5',1208, min_images_per_label=5)
    clr.prepare()
    clr.train()
    clr.evaluation()