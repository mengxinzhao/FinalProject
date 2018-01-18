#!/usr/bin/env python

## this file implements accuracy rate, false Acceptance rate, false reject rate, equal errorrate
## weighted precision score, recall score and plot ROC
## For biometrics performance I use
## https://precisebiometrics.com/wp-content/uploads/2014/11/White-Paper-Understanding-Biometric-Performance-Evaluation.pdf
## https://security.stackexchange.com/questions/57589/determining-the-accuracy-of-a-biometric-system

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve,auc,precision_score,recall_score,average_precision_score
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def accuracy(predictions, true_labels):
    """
    :param prediction:  N x num_class probability array
    :param grand_truth: N array of labels.
    :return: true prediction / all prediction
    """
    predicted_labels = np.argmax(predictions, axis=1)
    return np.mean(np.equal(predicted_labels,true_labels))

def false_accept(predictions,true_labels, threshold=0.2):
    """
    :param predictions: prediction array  N x num_class probability array
    :param true_labels: true test label array N array of labels.
    :param threshold:
    :return: number of imposter scores exceeding threshold / number of all imposters
    """
    # for each test, only one is a genuine everyone is seem as imposter
    num_imposter = (len(np.unique(true_labels)) - 1 )* len(predictions)
    thresholded = predictions  - threshold

    num_accept = 0
    for test_id in range(len(thresholded)):
        num_accept +=  np.sum(thresholded[test_id]>0 )
        if thresholded[test_id, true_labels[test_id]] > 0:
            num_accept -=1 # minus  true acceptance

    return num_accept /num_imposter

def false_reject(predictions,true_labels, threshold=0.2):
    """
    :param predictions: prediction array
    :param true_labels: true test label array
    :param threshold:
    :return: number of genuines scores below threshold/number of all genuine scores
    """
    num_genuine = len(true_labels)
    thresholded = predictions - threshold

    return np.sum([1 for test_id in range(len(predictions)) if thresholded[test_id, true_labels[test_id]] < 0 ])/num_genuine

# TODO: there has to be a better way to draw the curve and estimate the ERR
def det_curve(prediction,true_labels):
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
        far.append(false_accept(predictions,true_labels,thres_value))
        frr.append(false_reject(predictions,true_labels,thres_value))
    far = np.vstack(far)
    frr = np.vstack(frr)
    thres = np.vstack(thres)
    return far, frr, thres

def det_curve_plot(prediction,true_labels):

    far,frr,thres = det_curve(prediction,true_labels)
    plt.figure()
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

def eer(predcition,true_labels):
    """

    :param predcition:
    :param true_labels:
    :return:  equal error rate when false acceptance and false rejection rate are minimal and optimal, thres
    """
    far,frr,thres = det_curve(predictions,true_labels)
    idx = np.nanargmin(np.absolute(far - frr))
    return  far[idx], thres[idx]

def convert_prediction_to_one_against_rest_encode(predictions):
    """

    :param predictions:
    :return: one against rest encoded prediction matrix
    """
    test_scores = np.zeros_like(predictions)
    # binarize the predictions
    pos = np.argmax(predictions,axis=1)
    for test_id in range(len(predictions)):
            test_scores[test_id,pos[test_id]] = 1
    return test_scores

def convert_labels_to_one_against_rest_encode(grand_truth):

    truth_scores = np.zeros_like(predictions)
    # change it back to one against all encoding
    for test_id in range(len(truth_scores)):
        truth_scores[test_id, grand_truth[test_id]] = 1
    return truth_scores

def weighted_precision(predictions,grand_truth,num_class):
    labels = np.eye(num_class)
    truth_scores = convert_labels_to_one_against_rest_encode(grand_truth)
    test_scores = convert_prediction_to_one_against_rest_encode(predictions)
    return precision_score(truth_scores,test_scores,labels=np.unique(truth_scores),average='weighted')

def weighted_recall(predictions,grand_truth,num_class):
    labels = np.eye(num_class)
    truth_scores = convert_labels_to_one_against_rest_encode(grand_truth)
    test_scores = convert_prediction_to_one_against_rest_encode(predictions)
    return recall_score(truth_scores,test_scores,labels=np.unique(truth_scores),average='weighted')

def multiclass_roc_plot(predictions, grand_truth, thres= 0.2):
    # codes from sklearn

    # Compute ROC curve and ROC area for each class
    truth_score = convert_labels_to_one_against_rest_encode(grand_truth)
    test_scores = convert_prediction_to_one_against_rest_encode(predictions)
    lw = 2
    n_classes = len(np.unique(grand_truth))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_scores[:, i], truth_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_scores.ravel(), truth_score.ravel(),pos_label=1)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=lw)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=lw)

    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    #for i, color in zip(range(n_classes), colors):
    #    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #                   ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    num_class = 4
    num_test = 10
    # prediction for num_class. row sums up to 1
    predictions = np.random.rand(num_test, num_class)
    for row in range(len(predictions)):
        predictions[row] = predictions[row]/predictions[row].sum()
    print("prediction matrix\n", predictions)
    print("predicted labels ", np.argmax(predictions,axis=1))
    # grand truth 10 class
    grand_truth = np.random.choice(num_class,num_test)
    print("grand truth",grand_truth)
    print("accuracy: {0:.4f}".format(accuracy(predictions,grand_truth)))
    print("false_accept: {0:.4f}".format(false_accept(predictions, grand_truth)))
    print("false_reject: {0:.4f}".format(false_reject(predictions, grand_truth)))
    det_curve_plot(predictions,grand_truth)
    errate , thres= eer(predictions, grand_truth)
    print("eer: ",errate)
    print("thres: ",thres )
    multiclass_roc_plot(predictions, grand_truth)
    print("weighted precision: ", weighted_precision(predictions,grand_truth,num_class))
    print("weighted recall: ", weighted_recall(predictions, grand_truth,num_class))
