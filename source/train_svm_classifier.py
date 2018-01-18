#!/usr/bin/env python
# this file trains a SVM probability classifier

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score
import logging

from evaluate_classifier import accuracy, multiclass_roc_plot
from bottleneck_features import Features

logging.basicConfig(filename='./svm.log',level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SVMClassifier():
    def __init__(self,dataset_path, classfier_filename,num_class ):
        self.dataset_path = dataset_path

        self.feature = Features(self.dataset_path, 224, 224, num_class,face_crop=False, features_dir='../data/')

        self.train_codes,self. train_labels, self. test_codes,self. test_labels = \
            self.feature.get_train_test_set('../data/bottleneck_features.npy', '../data/labels.npy')

        # train_codes is 4 dimentions(num_samples, 1, 1, features). Need remove the middle 2 dimentions
        self.train_codes = np.squeeze(self.train_codes)
        self.test_codes = np.squeeze(self.test_codes)
        # label codes are binarized. Here SVM requests to convert back to numerical
        self.train_labels = np.argmax(self.train_labels, axis=1)
        self.test_labels = np.argmax(self.test_labels,axis=1)
        self.num_class = num_class
        self.model = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-4,
                    C=1.0, multi_class='ovr', fit_intercept=True,
                    intercept_scaling=1, class_weight=None, verbose=False,
                    random_state=None, max_iter=1000)

        self.classifier_filename = classfier_filename

    def train(self):

        self.model.fit(self.train_codes, self.train_labels)
        joblib.dump(self.model, self.classifier_filename)

        logger.info('Saved classifier model to file "%s"' % self.classifier_filename)


    def evaluate(self):
        loaded_model = joblib.load(self.classifier_filename)
        predictions = loaded_model.predict(self.test_codes)
        np.save('../model/svm_prediction.npy',predictions)

        logger.info("accuracy score {:.4f}".format(loaded_model.score(self.test_codes, self.test_labels)))
        test_idcs = np.load('../model/svm_test_labels.npy')
        # what went wrong
        for test_id in range(len(self.test_labels)):
            if predictions[test_id] != self.test_labels[test_id]:
                ## label number starts from 1 in the colorferet
                logger.info("data id  {} predicted label {}, true label {}".format(test_idcs[test_id], predictions[test_id]+1,self.test_labels[test_id]+1))
        logger.info("weighted precision score {:.4f}".format(precision_score(self.test_labels,predictions,average='weighted')))
        logger.info("weighted recall score {:.4f}".format(recall_score(self.test_labels,predictions,average='weighted')))
        multiclass_roc_plot(predictions,self.test_labels,self.num_class)



if __name__ == '__main__':
    clr = SVMClassifier('/Volumes/ML/ColorFeret_Test/','../model/svm_model.npy',1208)
    clr.train()
    clr.evaluate()

