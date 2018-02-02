#!/usr/bin/env python

## this file implements accuracy rate, false Acceptance rate, false reject rate, equal errorrate
## weighted precision score, recall score and plot ROC
## For biometrics performance I use
## https://precisebiometrics.com/wp-content/uploads/2014/11/White-Paper-Understanding-Biometric-Performance-Evaluation.pdf
## https://security.stackexchange.com/questions/57589/determining-the-accuracy-of-a-biometric-system

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve,auc,precision_score,recall_score,average_precision_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import logging
from collections import OrderedDict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# preserving order
def unique_labels(labels):
    labels = np.squeeze(labels)
    uniq = OrderedDict()
    for i in labels:
         uniq[i] = 1
    return uniq.keys()

def accuracy(predict_probs, true_labels):
    """
    :param prediction:  prediction array  N x num_class probability array
    :param true_labels: N array of labels.might be (N,) dimesion.
    :return: true prediction / all prediction
    """
    true_labels = np.squeeze(true_labels)
    label_set = np.unique(true_labels)
    predicted_labels = [label_set[id] if id < len(label_set) else -1  for id in np.argmax(predict_probs,axis = 1)]
    return np.mean(np.equal(predicted_labels,true_labels))

def false_accept(predict_probs,true_labels, threshold=0.2):
    """
    :param predictions: prediction array  N x num_class probability array
    :param true_labels: true test label array N array of labels.
    :param threshold:
    :return: number of imposter scores exceeding threshold / number of all imposters
    """
    true_labels = np.squeeze(true_labels)
    label_to_index = dict()
    for i,label in enumerate(np.unique(true_labels)):
        label_to_index[label] = i
    # for each test, only one is a genuine everyone is seem as imposter
    num_imposter = (len(np.unique(true_labels)) - 1 )* len(predict_probs)
    thresholded = predict_probs  - threshold

    num_accept = 0
    for test_id in range(len(thresholded)):
        num_accept +=  np.sum(thresholded[test_id]>0 )
        if true_labels[test_id] in label_to_index.keys():
            if thresholded[test_id, label_to_index[true_labels[test_id]]] > 0:
                num_accept -=1 # minus  true acceptance
        else:
            logger.warn("predicted label {} not seen in the test label set!".format(true_labels[test_id]))

    return num_accept /num_imposter

def false_reject(predict_probs,true_labels, threshold=0.2):
    """
    :param predictions: prediction array
    :param true_labels: true test label array
    :param threshold:
    :return: number of genuines scores below threshold/number of all genuine scores
    """
    true_labels = np.squeeze(true_labels)
    label_to_index = dict()
    for i,label in enumerate(np.unique(true_labels)):
        label_to_index[label] = i
    num_genuine = len(true_labels)
    thresholded = predict_probs - threshold

    return np.sum([1 for test_id in range(len(predict_probs))
                   if true_labels[test_id] in label_to_index.keys()
                   and thresholded[test_id, label_to_index[true_labels[test_id]]] < 0 ]
                  )/num_genuine

# TODO: there has to be a better way to draw the curve and estimate the ERR
def det_curve(predict_probs,true_labels):
    """
    :param prediction:  prediction array
    :param true_labels: true test label array
    :return: false_accept rate, false_reject rate , threshold
    https://www.webopedia.com/TERM/E/equal_error_rate.html
    """
    thres = []
    far = []
    frr = []
    values = np.linspace(0.001,1,100)

    for thres_value in values:
        thres.append(thres_value)
        far.append(false_accept(predict_probs,true_labels,thres_value))
        frr.append(false_reject(predict_probs,true_labels,thres_value))
    far = np.vstack(far)
    frr = np.vstack(frr)
    thres = np.vstack(thres)
    return far, frr, thres

def det_curve_plot(predict_probs,true_labels):

    far,frr,thres = det_curve(predict_probs,true_labels)
    plt.figure(figsize=(5, 5),)
    plt.plot(far, frr, label='DET')
    plt.plot(thres,frr,  label='FRR')
    plt.plot(thres, far, label='FAR')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Reject Rate')
    plt.title('DET curve')
    plt.legend(loc="lower right")
    plt.show()
    return far,frr,thres

def equal_error_rate(predict_probs,true_labels):
    """

    :param predcition:
    :param true_labels:
    :return:  equal error rate when false acceptance and false rejection rate are minimal and optimal, thres
    """
    far,frr,thres = det_curve(predict_probs,true_labels)
    idx = np.nanargmin(np.absolute(far - frr))
    return  far[idx], thres[idx]

def binarize_labels(labels,num_class):

    true_labels = np.squeeze(labels)
    label_to_index = dict()
    for i,label in enumerate(np.unique(true_labels)):
        label_to_index[label] =i

    b_labels = np.zeros((len(labels),num_class))

    for test_id in range(len(labels)):
        b_labels[test_id, label_to_index[labels[test_id]]] = 1
    return b_labels

def multiclass_roc_plot(predicted_labels, true_labels, num_class):
    # codes  reference from sklearn

    # Compute ROC curve and ROC area for each class
    truth_scores = binarize_labels(np.squeeze(true_labels),num_class)
    test_scores =  binarize_labels(predicted_labels,num_class)
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_class):
        _fpr, _tpr, _ = roc_curve(test_scores[:, i], truth_scores[:, i])
        # have to exclude tpr none exist condition. that probably means there is no tp predicted at all
        if np.any(np.isnan(_fpr)) == False and np.any(np.isnan(_tpr)) == False:
            fpr[i],tpr[i] =  _fpr, _tpr
            roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_scores.ravel(), truth_scores.ravel(),pos_label=1)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    #all_fpr = np.unique(np.concatenate([fpr[i] for i in range(min(num_class,len(fpr)))]))
    all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr.keys()]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    actual_num_class = 0
    for i in fpr.keys():
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        actual_num_class +=1

    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(5, 5),)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=lw)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=lw)

    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    #for i, color in zip(range(3), colors):
    #    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #                   ''.format(i, roc_auc[i]))
    #                   ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

def test_metrics(predict_probs,true_labels,num_class):
    print("accuracy: {0:.4f}".format(accuracy(predict_probs,true_labels)))
    print("false_accept: {0:.4f}".format(false_accept(predict_probs, true_labels)))
    print("false_reject: {0:.4f}".format(false_reject(predict_probs, true_labels)))
    det_curve_plot(predict_probs,true_labels)
    errate , thres= equal_error_rate(predict_probs, true_labels)
    print("equal error rate: ",errate)
    print("threshold: ",thres )
    predicted_labels = np.argmax(predict_probs,axis=1)
    multiclass_roc_plot(predicted_labels, true_labels,num_class)

if __name__ == '__main__':
    num_class = 15
    num_test =  50

    probs = np.random.rand(num_test, num_class)
    probs = normalize(probs, axis=1, norm='l1')
    # grand truth 10 class
    grand_truth = np.random.choice(num_class,num_test)
    #print("probility matrix:",probs)
    print("grand truth:",grand_truth)
    print("predicted labels:",np.argmax(probs,axis=1))

    test_metrics(probs,grand_truth,num_class)

    predictions = np.load('../model/svm_prediction_2.npy')
    logger.info("prediction shape: {}".format(predictions.shape))
    predict_probs = np.load('../model/svm_prediction_probability_2.npy')
    test_label = np.squeeze(np.load('../model/svm_test_labels_2.npy'))
    test_metrics(predict_probs,test_label,len(np.unique(test_label)))



