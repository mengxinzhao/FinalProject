#!/usr/bin/env python

import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense,Input, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import  SGD
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

batch_num = 32
alpha = 0.2

def triplet_loss(y_true, y_pred):
    # L= ||anchor_embedding - positive_embedding||  + alpha - ||anchor_embedding - negative_embedding||
    y_pred = K.l2_normalize(y_pred,axis=1)
    batch = batch_num
    ref1 = y_pred[0:batch,:]
    pos1 = y_pred[batch:batch+batch,:]
    neg1 = y_pred[batch+batch:3*batch,:]
    dis_pos = K.sqrt(K.sum(K.square(ref1 - pos1), axis=1, keepdims=True))
    dis_neg = K.sqrt(K.sum(K.square(ref1 - neg1), axis=1, keepdims=True))
    alpha = 0.2
    margin = K.constant(alpha)
    print(y_pred)
    print(K.mean(K.maximum(0.0,dis_pos-dis_neg+margin)))
    return K.mean(K.maximum(0.0,dis_pos-dis_neg+margin))

def identity_accurancy(y_true, y_pred):
    # L =  <y_pred, y_pred.T> inner product
    # the bigger the better
    y_pred = K.l2_normalize(y_pred, axis=1)
    pred = y_pred[0:batch_num, :]
    print("identity_distance:", pred)
    return K.mean(K.sqrt(K.sum(K.square(pred), axis=1, keepdims=True)))

def generate_anchor_positive(ids):
    """
    :param ids: index array of the same identity
    :return: all pairs of [anchor, positive]. position matters
    """
    if len(ids)<2:
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

def generate_anchor_negative(x_codes, labels,pairs,model):
    """
    :param x_codes: training codes
    :param labels: training labels
    :param pairs:  [anchor, positive]
    :param model: model to get embeddings
    :return:  triplets index for  [anchor, positive, negative]
    """
    ## random sample pair length of negative samples and get the embeddings
    triplets = []
    for idx in range(len(pairs)):
        # make it 4 dimension(1,1,1,2048) for resnet prediction
        anchor_embeddings = model.predict(np.expand_dims(x_codes[pairs[idx,0]],axis=0))
        positive_embeddings = model.predict(np.expand_dims(x_codes[pairs[idx,1]],axis=0))
        ng_ids, = np.where((labels != labels[pairs[idx, 0]]).any(axis=1))
        num_tries = 0
        while True:
            ## what if it can't be found?
            ng_id = random.choice(ng_ids)
            negative_embedding = model.predict(np.expand_dims(x_codes[ng_id],axis=0))
            dis_pos = np.linalg.norm(anchor_embeddings - positive_embeddings) # L2 norm
            dis_neg = np.linalg.norm(anchor_embeddings - negative_embedding)
            if dis_pos < dis_neg:
                triplets.append(np.array([pairs[idx, 0], pairs[idx, 1], ng_id]))
                break
            num_tries +=1
            # this one probably won't have any good patch
            if num_tries>=1000:
                print("no semi negative match for anchor indx: ",pairs[idx,0]," will randomly choose one")
                ng_id = random.choice(ng_ids)
                triplets.append(np.array([pairs[idx,0],pairs[idx,1],ng_id]))
                break

    return np.array(triplets)



def get_next_triplet_batch(x_codes,labels,embedding_layer, batch_num = 32):
    """
    :param x_codes: training codes
    :param labels: labels
    :param model: the current cnn model
    :return:  vstacked index array anchor ,positive ,negative

    """
    # google's facenet online triplet generator
    # use all anchor-positive pairs
    # semi hard negative that  ||anchor_embedding - positive_embedding|| < ||anchor_embedding - negative_embedding||
    # since the x_codes and labels are already shuffed and randomized, will just iterate through all ids
    pairs = []
    triplets = []
    yield_idx = 0
    batch_idx = yield_idx
    unique_labels = np.unique(labels,axis=0)
    for ID in unique_labels:
        ids, = np.where((labels==ID).all(axis=1)) # return (row_idx_array,)
        # get anchor-positive pairs.
        if len(pairs) >0:
            pairs = np.vstack((pairs,generate_anchor_positive(ids)))
        else:
            pairs = generate_anchor_positive(ids)
        # generate enough anchor-positive pairs to form a batch
        if pairs.shape[0] - yield_idx >= batch_num or (ID == unique_labels[-1]).all():
            batch_idx = pairs.shape[0] - 1
        else:
            continue

        while(yield_idx + batch_num-1 <=batch_idx):
            triplets= generate_anchor_negative(x_codes,labels, pairs[yield_idx:yield_idx + batch_num],embedding_layer)
            yield triplets[:, 0],triplets[:, 1],triplets[:, 2]
            yield_idx += batch_num

        # the last batch
        if (ID == unique_labels[-1]).all() and yield_idx <= batch_idx:
            triplets = generate_anchor_negative(labels, pairs[yield_idx:batch_idx + 1],embedding_layer)
            yield triplets[:, 0],triplets[:, 1],triplets[:, 2]
            yield_idx = batch_idx


class CNNClassifier():
    def __init__(self,dataset_path, classfier_filename,num_class,min_images_per_label=5,use_triplet_loss=False ):
        self.feature = Features(dataset_path, 224, 224, num_class,
                                face_crop=False, min_images_per_label = min_images_per_label, features_dir='../data/')

        #self.num_class = num_class  # almost not important
        self.use_triplet_loss = use_triplet_loss
        self.classifier_filename = classfier_filename
        self.num_embeddings = 128
        self.lb = LabelBinarizer()
        self.alpha = 0.2
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
            self.model.add(Dense(self.num_embeddings, kernel_initializer='random_uniform',
                bias_initializer='zeros',name='embeddings'))
            self.model.add(BatchNormalization(name='output'))
            ## Todo: get the optimizer setting right
            self.model.compile(loss={'output':triplet_loss},
                                optimizer=SGD(lr=0.0001,momentum=0.9, decay=0.0005),
                                metrics={'output':identity_accurancy})
            self.model.summary()

    def train(self):

        ## split training set further for training and validation set

        start_time = time.time()
        logger.info('Training start...' )
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
            ## Todo: finish the training and evalution
            epochs = 1
            evaluate_every = 50
            for i in range(epochs):
                iter = 0;
                for batch in get_next_triplet_batch(self.x_codes,self.y_labels,self.model):
                    tpl_batch = np.ravel(batch)

                    # we are not about prediction.
                    # For the anchor,positive samples, all should be learned as recoginzed.
                    # The negative is all 0 (since the metrics doesn't look at any prediciton beyond batch_num
                    # we can set the y_lables to all 1
                    # loss is list of [triplet_loss, identity_accurancy]
                    loss = self.model.train_on_batch(self.x_codes[tpl_batch], np.ones((len(tpl_batch))))
                    if iter % 5 == 0:
                        print("epoch ",i,"iteration ", iter, " ==> [training loss, identity accurancy]: ", loss)
                    iter+=1
            return 0

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
    clr = CNNClassifier('/Volumes/ML/ColorFeret_Test/','../model/weights_triplet_loss.hdf5',1208,
                        min_images_per_label=5,use_triplet_loss=True)
    clr.prepare()
    clr.train()
    #clr.evaluation()
