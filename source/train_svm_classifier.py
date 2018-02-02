#!/usr/bin/env python
# this file trains a SVM probability classifier

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import logging
import time

from evaluate_classifier import accuracy, multiclass_roc_plot,false_accept, false_reject,equal_error_rate,det_curve_plot
from bottleneck_features import Features

logging.basicConfig(filename='./svm.log',level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SVMClassifier():
    def __init__(self,dataset_path, classfier_filename,num_class,min_images_per_label=5 ):
        self.feature = Features(dataset_path, 224, 224, num_class,
                                face_crop=False, min_images_per_label = min_images_per_label, features_dir='../data/')

        self.num_class = num_class
        svc = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-4,
                    C=1.0, multi_class='ovr', fit_intercept=True,
                    intercept_scaling=1, class_weight=None, verbose=False,
                    random_state=None, max_iter=1000)
        self.model = CalibratedClassifierCV(svc, cv=3, method='sigmoid')
        self.classifier_filename = classfier_filename


    def prepare(self):
        self.train_codes = np.squeeze(np.load('../data/train_codes.npy'))
        self.test_codes = np.squeeze(np.load('../data/test_codes.npy'))
        self.train_labels = np.squeeze(np.load('../data/train_labels.npy'))
        self.test_labels = np.squeeze(np.load('../data/test_labels.npy'))

    def train(self):
        start_time = time.time()
        logger.info('Training start...' )
        self.model.fit(self.train_codes, self.train_labels)
        joblib.dump(self.model, self.classifier_filename)
        logger.info('Completed in {} seconds'.format(time.time() - start_time))
        logger.info('Saved classifier model to file "%s"' % self.classifier_filename)

    def evaluate(self):
        loaded_model = joblib.load(self.classifier_filename)
        predictions = loaded_model.predict(self.test_codes)
        np.save('../model/svm_prediction.npy',predictions)

        predict_probs = loaded_model.predict_proba(self.test_codes)
        np.save('../model/svm_prediction_proba.npy', predictions)

        logger.info("accuracy score {:.4f}".format(loaded_model.score(self.test_codes, self.test_labels)))
        test_idcs = np.load('../data/test_labels_idcs.npy')
        # what went wrong
        for test_id in range(len(self.test_labels)):
            if predictions[test_id] != self.test_labels[test_id]:
                ## label number starts from 1 in the colorferet
                logger.info("data id  {} predicted label {}, true label {}".format(test_idcs[test_id], predictions[test_id]+1,self.test_labels[test_id]+1))
        logger.info("weighted precision score {:.4f}".format(precision_score(self.test_labels,predictions,average='weighted')))
        logger.info("weighted recall score {:.4f}".format(recall_score(self.test_labels,predictions,average='weighted')))
        print("accuracy: {0:.4f}".format(accuracy(predict_probs, self.test_labels)))
        print("false_accept: {0:.4f}".format(false_accept(predict_probs, self.test_labels)))
        print("false_reject: {0:.4f}".format(false_reject(predict_probs, self.test_labels)))
        det_curve_plot(predict_probs, self.test_labels)
        errate, thres = equal_error_rate(predict_probs, self.test_labels)
        print("equal error rate: ", errate)
        print("threshold: ", thres)

        #multiclass_roc_plot(predictions,self.test_labels,len(np.unique(self.test_labels)))

if __name__ == '__main__':
    clr = SVMClassifier('/Volumes/ML/ColorFeret_Test/','../model/svm_model.npy',1208, min_images_per_label=5)
    clr.prepare()
    clr.train()
    clr.evaluate()

