#!/usr/bin/env python

import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense,Input, BatchNormalization,Lambda
from keras.models import Sequential, Model
from keras.optimizers import  SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import  EarlyStopping
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.regularizers import  l2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from evaluate_classifier import accuracy, multiclass_roc_plot,false_accept, false_reject,equal_error_rate,det_curve_plot
from bottleneck_features import Features
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

batch_num = 128
alpha = 0.2

def triplet_loss(y_true, y_pred):
    # L= ||anchor_embedding - positive_embedding||  + alpha - ||anchor_embedding - negative_embedding||

    #y_pred = K.l2_normalize(y_pred,axis=1)
    batch = batch_num
    anchor = y_pred[0:batch,:]
    pos = y_pred[batch:batch+batch,:]
    neg = y_pred[batch+batch:3*batch,:]
    dis_pos = K.sqrt(K.sum(K.square(anchor - pos), axis=1, keepdims=True))
    dis_neg = K.sqrt(K.sum(K.square(anchor - neg), axis=1, keepdims=True))
    alpha = 0.6  # the paper says 0.2 seems to too easy for this small data set
    margin = K.constant(alpha)
    return K.mean(K.maximum(0.0,dis_pos-dis_neg+margin))

def identity_accurancy(y_true, y_pred):
    # L =  ||anchor_embedding - negative_embedding|| - ||anchor_embedding - positive_embedding||
    # the bigger the better
    #y_pred = K.l2_normalize(y_pred, axis=1)
    anchor = y_pred[0:batch_num, :]
    pos = y_pred[batch_num:2*batch_num,:]
    neg = y_pred[2*batch_num:3*batch_num,:]
    dis_pos = K.sqrt(K.sum(K.square(anchor - pos), axis=1, keepdims=True))
    dis_neg = K.sqrt(K.sum(K.square(anchor - neg), axis=1, keepdims=True))
    return K.mean(K.maximum(0.0, dis_neg - dis_pos ))


class tripletDataGenerator(object):
    "data generator for keras"
    def __init__(self,batch_size, shuffle = False):
        self.batch_size = batch_size
        self.shuffle = shuffle


    def generate_anchor_positive(self,ids):
        """
        :param ids: index array of the same identity
        :return: all pairs of [anchor, positive]. position matters
        """
        if len(ids)<2:
            #logger.warn("identity face pictures less than 2 . no anchor positive pair")
            return []
        results = []
        datasets=set()
        datasets.add(ids[0])
        datasets.add(ids[1])
        results.append(np.array([ids[0],ids[1]]))
        results.append(np.array([ids[1], ids[0]]))
        for idx in range(2,len(ids)):
            for elem in datasets:
              results.append(np.array([ids[idx],elem]))
              results.append(np.array([elem,ids[idx]]))
            datasets.add(ids[idx])
        return np.array(results)

    def generate_anchor_negative(self,x_codes, labels, pairs, model):
        """
        :param x_codes: training codes
        :param labels: training labels
        :param pairs:  [anchor, positive]
        :param model: model to get embeddings
        :return:  triplets index for  [anchor, positive, negative]
        """
        ## random sample pair length of negative samples and get the embeddings
        triplets = []
        ## WA have to run it in tensorflow otherwise complaining model not found
        # https: // github.com / keras - team / keras / issues / 2397
        with graph.as_default():
            for idx in range(len(pairs)):
                # make it 4 dimension(1,1,1,2048) for resnet prediction
                model._make_predict_function()
                anchor_embeddings = model.predict(np.expand_dims(x_codes[pairs[idx, 0]], axis=0))
                #print(anchor_embeddings)
                positive_embeddings = model.predict(np.expand_dims(x_codes[pairs[idx, 1]], axis=0))
                ng_ids, = np.where((labels != labels[pairs[idx, 0]]).any(axis=1))
                num_tries = 0
                while True:
                    ## what if it can't be found?
                    ng_id = random.choice(ng_ids)
                    negative_embedding = model.predict(np.expand_dims(x_codes[ng_id], axis=0))
                    dis_pos = np.linalg.norm(anchor_embeddings - positive_embeddings)  # L2 norm
                    dis_neg = np.linalg.norm(anchor_embeddings - negative_embedding)
                    if dis_pos < dis_neg:
                        triplets.append(np.array([pairs[idx, 0], pairs[idx, 1], ng_id]))
                        break
                    num_tries += 1
                    # this one probably won't have any good patch
                    if num_tries >= 1000:
                        logger.warn("no semi negative match for anchor index: {} , will randomly choose one".format(pairs[idx, 0]))
                        ng_id = random.choice(ng_ids)
                        triplets.append(np.array([pairs[idx, 0], pairs[idx, 1], ng_id]))
                        break

            return np.array(triplets)

    def get_unique_labels(self,labels):
        _, idx = np.unique(labels,  axis=0,return_index=True)
        return labels[np.sort(idx)]

    def generate(self, data ,labels, embedding_layer ):
        "generate batch of sample"
        x_codes = data
        labels = labels

        while True:
            if self.shuffle == True:
                indics = np.arange(len(labels))
                np.random.shuffle(indics)
                x_codes = x_codes[indics]
                labels = labels[indics]

            pairs = []
            triplets = []
            yield_idx = 0
            batch_idx = yield_idx
            # unique index needs to presever otherwise it will be always sorted from small to big
            unique_labels = self.get_unique_labels(labels)
            batch_num = self.batch_size
            # google's facenet online triplet generator
            # use all anchor-positive pairs
            # semi hard negative that  ||anchor_embedding - positive_embedding|| < ||anchor_embedding - negative_embedding||
            # since the x_codes and labels are already shuffed and randomized, will just iterate through all ids
            for ID in unique_labels:
                ids, = np.where((labels == ID).all(axis=1))  # return (row_idx_array,)
                # get anchor-positive pairs.
                ap_pair = self.generate_anchor_positive(ids)
                if len(pairs) > 0 and len(ap_pair) > 0:
                    pairs = np.vstack((pairs, ap_pair))
                elif len(ap_pair) > 0:
                    pairs = ap_pair

                if len(pairs) - yield_idx >= batch_num : #or (ID == unique_labels[-1]).all():
                    batch_idx = len(pairs)- 1
                else:
                    continue
                while (yield_idx + batch_num - 1 <= batch_idx):
                    triplets = self.generate_anchor_negative(x_codes, labels, pairs[yield_idx:yield_idx + batch_num],embedding_layer)
                    tpl_batch = np.ravel((triplets[:, 0], triplets[:, 1], triplets[:, 2]))
                    # we are not about prediction.
                    # For the anchor,positive samples, all should be learned as recoginzed.
                    # The negative is all 0 (since the metrics doesn't look at any prediciton beyond batch_num
                    # we can set the y_lables to all 1
                    yield x_codes[tpl_batch], np.ones(len(tpl_batch))
                    yield_idx += batch_num

            # the last batch is discarded it doesn't fit batch size. tensor flow complains.
            # each time at the beginning we shuffle so the order of picking up ID is different everytime.



class CNNClassifier():
    def __init__(self,dataset_path, classfier_filename,min_images_per_label=5,use_triplet_loss=False ):
        self.feature = Features(dataset_path, 224, 224, face_crop=False, min_images_per_label = min_images_per_label, features_dir='../data/')
        self.use_triplet_loss = use_triplet_loss
        self.classifier_filename = classfier_filename
        self.num_embeddings = 128
        self.lb = LabelBinarizer()
        self.alpha = 0.2
        K.set_image_data_format('channels_last')


    def prepare(self):
        labels = np.load('../data/labels_10min.npy')
        self.train_codes = np.load('../data/lfw_train_codes.npy')
        self.test_codes = np.load('../data/lfw_test_codes.npy')
        self.train_labels = np.squeeze(np.load('../data/lfw_train_labels.npy'))
        self.test_labels = np.squeeze(np.load('../data/lfw_test_labels.npy'))

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

        # classifier
        logger.info("train codes shape {}".format(self.x_codes.shape))
        logger.info("validation codes shape {}".format(self.val_codes.shape))
        logger.info("train label shape {}".format(self.y_labels.shape))
        logger.info("validation label shape {}".format(self.val_labels.shape))

        if not self.use_triplet_loss:

            self.model = Sequential()
            self.model.add(GlobalAveragePooling2D(name='global_average', input_shape=self.x_codes.shape[1:]))
            self.model.add(Dense(len(np.unique(self.test_labels)), activation='softmax', name='prediction'))
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            self.model.summary()
        else:
            # implement triplet lose

            self.model = Sequential()
            self.model.add(GlobalAveragePooling2D(name='global_average', input_shape=self.x_codes.shape[1:]))
            self.model.add(Dense(self.num_embeddings, kernel_initializer=TruncatedNormal(stddev=0.001),
                bias_initializer='zeros',name='embeddings',kernel_regularizer=l2(0.01)))
            self.model.add(Lambda(lambda  x: K.l2_normalize(x,axis=1),name='output'))
            self.model.compile(loss={'output':triplet_loss},
                                optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                                metrics={'output':identity_accurancy})
            self.model.summary()
            global graph
            graph = tf.get_default_graph()

    def get_triplet_sample_numbers (self,labels):
        "get triplet sample numberss"
        unique_labels = np.unique(labels, axis=0)
        sum = 0
        for l in unique_labels:
            row, = np.where((l == labels).all(axis=1))
            if len(row) > 1:
                sum += len(row) * (len(row) - 1)
        return sum

    def train(self):

        ## split training set further for training and validation set

        start_time = time.time()
        logger.info('Training starts...' )
        checkpointer = ModelCheckpoint(filepath=self.classifier_filename,
                                       verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=20)
        if not self.use_triplet_loss :
            evaluation = self.model.fit(self.x_codes, self.y_labels,
                            validation_data=(self.val_codes, self.val_labels),
                            epochs=1000, batch_size=batch_num, callbacks=[checkpointer,early_stopping], verbose=1)
            logger.info('Completed in {} seconds'.format(time.time() - start_time))
            logger.info("Training result {}".format(evaluation))
            pickle.dump(evaluation.history, open("../model/cnn_train_history.pickle", "wb"))
            return evaluation
        else:
            train_data_params = {
                'batch_size':batch_num,
                'shuffle':True
            }
            val_data_params = {
                'batch_size': batch_num,
                'shuffle': True
            }

            train_data_generator  = tripletDataGenerator(**train_data_params).generate(self.x_codes,self.y_labels,self.model)
            val_data_generator = tripletDataGenerator(**val_data_params).generate(self.val_codes,self.val_labels,self.model)
            # calculate triplet sample's total for train/test

            evaluation = self.model.fit_generator(train_data_generator,
                      steps_per_epoch = self.get_triplet_sample_numbers(self.y_labels)//batch_num ,
                      epochs=2,
                      verbose=1,
                      callbacks=[checkpointer,early_stopping],
                      validation_data=val_data_generator,
                      validation_steps = self.get_triplet_sample_numbers(self.val_labels)//batch_num ,
                      use_multiprocessing = False)
            logger.info('Completed in {} seconds'.format(time.time() - start_time))
            logger.info("Training result {}".format(evaluation))
            pickle.dump(evaluation.history, open("../model/cnn_train_history.pickle", "wb"))
            return 0

    def evaluation(self):
        # TODO evaluate the two methods using the same matrix

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
        test_idcs = np.load('../data/lfw_test_labels_idcs.npy')
        # what went wrong
        for test_id in range(len(self.test_labels)):
            if predictions[test_id] != self.test_labels[test_id]:
                logger.info(
                    "data id  {} predicted label {}, true label {}".format(test_idcs[test_id], predictions[test_id],
                                                                           self.test_labels[test_id] ))
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


if __name__=='__main__':
    clr = CNNClassifier('/Volumes/ML/lfw/','../model/weights_soft_max.hdf5',
                        min_images_per_label=10,use_triplet_loss=False)
    clr.prepare()
    clr.train()
    clr.evaluation()
