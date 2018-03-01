#!/usr/bin/env python

import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense,Input, BatchNormalization,Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.initializers import TruncatedNormal
from keras.regularizers import  l2,l1_l2,l1
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score,accuracy_score
import bob.measure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SoftmaxClassifier():
    def __init__(self,classfier_filename, data_dir):
        self.classifier_filename = classfier_filename
        self.lb = LabelBinarizer()
        self.batch_num = 128
        self.thres = 1.0
        self.data_dir = data_dir
        K.set_image_data_format('channels_last')

    def prepare(self):
        labels = np.load(os.path.join(self.data_dir, 'labels.npy'))
        self.train_codes = np.load(os.path.join(self.data_dir, 'train_codes.npy'))
        self.test_codes = np.load(os.path.join(self.data_dir, 'test_codes.npy'))
        self.train_labels = np.squeeze(np.load(os.path.join(self.data_dir, 'train_labels.npy')))
        self.test_labels = np.squeeze(np.load(os.path.join(self.data_dir, 'test_labels.npy')))
        self.test_idcs = np.load(os.path.join(self.data_dir,'test_labels_idcs.npy'))
        self.random_draw_codes = np.load(os.path.join(self.data_dir, 'random_draw_codes.npy'))
        self.random_draw_label = np.squeeze(np.load(os.path.join(self.data_dir, 'random_draw_labels.npy')))

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

        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D(name='global_average', input_shape=self.x_codes.shape[1:]))
        self.model.add(Dense(len(np.unique(self.test_labels)), activation='softmax', name='prediction'))
        # self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                           metrics=['accuracy'])

        self.model.summary()

    def train(self):

        ## split training set further for training and validation set

        start_time = time.time()
        logger.info('Training starts...' )
        checkpointer = ModelCheckpoint(filepath=self.classifier_filename,
                                       verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=20)
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True,
                                                 write_images=True,write_grads=True)
        tbCallBack.set_model(self.model)

        evaluation = self.model.fit(self.x_codes, self.y_labels,
                            validation_data=(self.val_codes, self.val_labels),
                            epochs=10000, batch_size=self.batch_num, callbacks=[checkpointer,early_stopping,tbCallBack], verbose=1)
        logger.info('Completed in {} seconds'.format(time.time() - start_time))
        logger.info("Training result {}".format(evaluation))
        pickle.dump(evaluation.history, open("../model/soft_max_train_history.pickle", "wb"))
        return evaluation

    def train_history_visual(self):

        # summarize history for accuracy
        with open("../model/soft_max_train_history.pickle", 'rb') as pickle_file:
            history = pickle.load(pickle_file)
        plt.figure(figsize=(6,3))
        plt.subplot(1, 2, 1)
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='lower right')
        #plt.show()
        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()

    def predict(self, test_codes, top_k = 5):
        predicts = []
        self.model.load_weights(self.classifier_filename)
        predict_probs = self.model.predict_proba(test_codes)
        for idx in range(len(predict_probs)):
             if predict_probs[idx].max()>= self.thres:  ## need to be optimized
                 predicts.append(self.lb.inverse_transform(np.reshape(predict_probs[idx],
                                                                      (-1, len(predict_probs[idx]))),threshold=0.5))
             else:
                 predicts.append(np.array([]))
        #return np.array(predicts)
        return np.array(predicts)

    def determin_threshold(self):
        """
        :return:  determin threshold
        """
        self.model.load_weights(self.classifier_filename)
        print(self.val_codes.shape)
        predict_probs = self.model.predict_proba(self.val_codes)
        max_probs = predict_probs.max(axis=1)
        return max_probs.mean()
        
    def evaluation(self):
        true_accept = 0.0
        false_accept = 0.0
        true_reject = 0.0
        false_reject = 0.0

        negative_scores = []
        positive_scores = []

        self.thres = self.determin_threshold()
        logger.info("threshold {}".format(self.thres))
        logger.info("Testing all authorized labels...")
        predictions_thres = self.predict(self.test_codes,top_k=1)
        
        all_test = 0.0
        label_count = {}
        errors_closeset_count = {}
        errors_openset_count = {}
        train_labels,counts = np.unique(self.train_labels,return_counts=True)
        for test_label,count in zip(train_labels,counts):
            label_count[test_label] = count
            errors_closeset_count[count] = 0
            errors_openset_count[count] = 0

            
        for test_id in range(len(self.test_labels)):
            if np.asarray(np.where(predictions_thres[test_id] == self.test_labels[test_id])).size == 0:
                logger.info("data id  {} false rejected as label {}, true label {}"
                            .format(self.test_idcs[test_id], predictions_thres[test_id], self.test_labels[test_id]))
                false_reject += 1.0
                # what id is mistaken for what
                errors_closeset_count[label_count[self.test_labels[test_id]] ] += 1
            else:
                true_accept += 1.0
            all_test += 1.0

        TR = true_accept / all_test
        FR = false_reject / all_test

        logger.info("true_acceptance: {0:.4f}".format(TR))
        logger.info("false_rejection: {0:.4f}".format(FR))

        # analysis

        logger.info("Testing random draw labels...")
        label_seen = 0
        for in_label in np.unique(self.train_labels):
            tmp = len(np.where(self.random_draw_label == in_label)[0])
            if tmp > 0:
                label_seen +=tmp
        logger.info("{} out of {} labels are authorized.".format(label_seen, self.random_draw_label.size))

        # not in the system
        predictions_thres = self.predict(self.random_draw_codes)

        true_accept = 0.0
        false_accept = 0.0
        true_reject = 0.0
        false_reject = 0.0
        all_test = 0.0
        # what went wrong
        for test_id in range(len(self.random_draw_codes)):
            if (self.random_draw_label[test_id] == self.train_labels).any() == True:
                # in the system
                if np.asarray(predictions_thres[test_id]).size == 0 or predictions_thres[test_id] != self.random_draw_label[test_id]:
                    false_reject += 1.0
                    logger.info("data id  {} false rejected as label {}, true label {}"
                                .format(test_id, predictions_thres[test_id], self.random_draw_label[test_id]))
                else:
                    true_accept += 1.0
                    logger.info("data id  {} accepted label  {} true label {} "
                                .format(test_id, predictions_thres[test_id], self.random_draw_label[test_id]))
            else:
                # not in the system
                if np.asarray(predictions_thres[test_id]).size > 0:
                    false_accept += 1.0
                    logger.info("data id  {} false accepted as label {}, none autherized label {}"
                                .format(test_id, predictions_thres[test_id], self.random_draw_label[test_id]))
                    errors_openset_count[label_count[self.test_labels[test_id]] ] += 1
                else:
                    true_reject += 1.0
                    logger.info("data id  {} rejected label {}, none autherized"
                                .format(test_id, self.random_draw_label[test_id]))
            all_test += 1.0

        # logger.info("false_accept: {0:.4f}".format(false_accept))
        # logger.info("true_accept: {0:.4f}".format(true_accept))
        # logger.info("false_reject: {0:.4f}".format(false_reject))
        # logger.info("true_reject: {0:.4f}".format(true_reject))


        FAR = false_accept / all_test
        TR = true_reject/all_test
        logger.info("true_rejection: {0:.4f}".format(TR))
        logger.info("false_acceptance: {0:.4f}".format(FAR))

        for key,value in list(errors_closeset_count.items()):
            if value == 0.0:
                del errors_closeset_count[key]

        for key,value in list(errors_openset_count.items()):
            if value == 0.0:
                del errors_openset_count[key]

        plt.subplot(1, 1, 1)
        plt.scatter(list(errors_closeset_count.keys()), list(errors_closeset_count.values()),marker='x')
        plt.scatter(list(errors_openset_count.keys()),list(errors_openset_count.values()), marker='.')
        plt.title('errors vs pictures per labels')
        plt.ylabel('error count')
        plt.xlabel('picture count per label')
        plt.legend(['closeset', 'openset'], loc='upper right')
        plt.show()

        self.test_labels = np.vstack((self.test_labels[:,None], self.random_draw_label[:,None]))
        self.test_codes = np.vstack((self.test_codes, self.random_draw_codes))

        # combine test_label and random draw
        predictions_thres = self.predict(self.test_codes)
        # what went wrong
        true_accept = 0.0
        false_accept = 0.0
        true_reject = 0.0
        false_reject = 0.0

        for test_id in range(len(self.test_labels)):
            if (self.test_labels[test_id]== self.train_labels).any() == True:
                # in the system
                if np.asarray(predictions_thres[test_id]).size == 0 :#or predictions_thres[test_id] != self.test_labels[test_id]:
                    false_reject += 1.0
                    logger.info("data id  {} false rejected as label {}, true label {}"
                                .format(test_id, predictions_thres[test_id], self.test_labels[test_id]))
                else:
                    true_accept += 1.0
                    logger.info("data id  {} accepted label  {} true label {} "
                                .format(test_id, predictions_thres[test_id],self.test_labels[test_id] ))
            else:
                # not in the system
                if np.asarray(predictions_thres[test_id]).size > 0:
                    false_accept += 1.0
                    logger.info("data id  {} false accepted as label {}, none autherized label {}"
                                .format(test_id, predictions_thres[test_id], self.test_labels[test_id]))
                else:
                    true_reject += 1.0
                    logger.info("data id  {} rejected label {}, none autherized"
                                .format(test_id, self.test_labels[test_id]))

        # logger.info("false_accept: {0:.4f}".format(false_accept))
        # logger.info("true_accept: {0:.4f}".format(true_accept))
        # logger.info("false_reject: {0:.4f}".format(false_reject))
        # logger.info("true_reject: {0:.4f}".format(true_reject))

        if false_accept +true_accept ==0 :
            FAR = 0.0
        else:
            FAR = false_accept / (false_accept +true_accept)
        if false_reject + true_reject == 0:
            FRR = 0.0
        else:
            FRR = false_reject / (false_reject + true_reject)

        logger.info("false_acceptance: {0:.4f}".format(FAR))
        logger.info("false_rejection: {0:.4f}".format(FRR))



if __name__=='__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='../data', dest='input_dir')
    parser.add_argument('--file-name', type=str, action='store', default='../model/weights_soft_max.hdf5', dest='file_name')

    args = parser.parse_args()

    if os.path.exists(args.input_dir) == False:
        logger.error("input {} doesn't exist!".format(args.input_dir))


    clr = SoftmaxClassifier(args.file_name,data_dir=args.input_dir)
    clr.prepare()
    clr.train()
    clr.train_history_visual()
    clr.evaluation()

