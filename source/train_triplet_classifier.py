#!/usr/bin/env python

import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import math
import itertools
from keras.layers import GlobalAveragePooling2D,Flatten
from keras.layers import Dense,Input,Lambda,concatenate,BatchNormalization,Dropout,Activation
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adam
from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
from keras.callbacks import EarlyStopping,Callback,TensorBoard,LearningRateScheduler
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.regularizers import l2,l1,l1_l2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score,accuracy_score
import tensorflow as tf
import bob.measure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

batch_num =  64
embedding_num  = 64
alpha = 0.2
embeddings = []
labels = []


def step_decay(epoch):
    """
    :param epoch: epoch number
    :return: learning rate schedule
    """
    initial_lrate = 0.0005
    drop = 0.9
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    if lrate < 0.00001:
        lrate = 0.00001
    return lrate

def triplet_loss(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: triplet loss
    """
    # L= ||anchor_embedding - positive_embedding||  + alpha - ||anchor_embedding - negative_embedding||

    anchor, pos, neg = y_pred[:,0:embedding_num],y_pred[:,embedding_num:2*embedding_num],y_pred[:,2*embedding_num:3*embedding_num]
    dist_pos = K.sum(K.square(anchor - pos), axis=-1)
    dist_neg = K.sum(K.square(anchor - neg), axis=-1)
    margin = K.constant(alpha)
    return K.mean(K.sum((K.maximum(0.0,dist_pos-dist_neg+margin) )))


def accuracy(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: percentage of triplets satisfying  ||anchor_embedding - positive_embedding|| + alpha<||anchor_embedding - negative_embedding||
    """
    anchor, pos, neg = y_pred[:, 0:embedding_num], y_pred[:, embedding_num:2 * embedding_num], y_pred[:,2 * embedding_num:3 * embedding_num]
    dist_pos = K.sum(K.square(anchor - pos), axis=1)
    dist_neg = K.sum(K.square(anchor - neg), axis=1)
    margin = K.constant(alpha)
    return K.sum(K.cast(K.less(dist_pos+ margin ,dist_neg),dtype='int32'))/K.shape(y_pred)[0]

def l2_distance (anchor, query):
    """
    l2 distance between anchor and query embeddings
    :param anchor:
    :param query:
    :return:
    """
    return np.sum(np.square(anchor - query),axis=1)

def l2_similarity(query_embedding,label_embedings):
    """

    :param query_embedding:
    :param labels_embedings:
    :return:  l2 similarity and idics array of the closet identities label_embbedings toward query's embeddings
    """
    dist = np.zeros(len(label_embedings))
    #dist = np.linalg.norm(label_embedings -query_embedding,axis=1)
    dist = l2_distance(query_embedding, label_embedings)
    sorted_idcs = sorted(range(len(dist)), key=lambda k: dist[k])
    similarity = dist[sorted_idcs]
    return similarity, np.array(sorted_idcs)


def query_top_match(query_embedding,anchor_embedings,anchor_labels, thres, top_k=5):
    """return top match at threshold
    :param query_embedding:
    :param labels_embedings:
    :param anchor_labels:
    :paran anchor_thres;:
    :param top_num: top k matches
    :return: matching labels top_k matches at threshold
    """
    dist, idcs = l2_similarity(query_embedding,anchor_embedings)

    dist_top_k_idcs = np.where(dist < thres)[0][0:top_k]
    #return labels[idcs[dist_top_k_idcs]], labels[idcs[0]]
    if np.asarray(dist_top_k_idcs).size >0:
        dist_top_k_labels = anchor_labels[idcs[dist_top_k_idcs]]
        unique_labels,first_idcs, counts = np.unique(dist_top_k_labels, return_counts=True,return_index=True)
        match_idx = np.where(counts==counts.max())

        if np.asarray(match_idx).size >=2:  # tied match
            # the first occuring one in the top_k
            match_idx = np.argsort(first_idcs[match_idx])
            match = unique_labels[match_idx[0]]
        else:
            match = unique_labels[match_idx]
        match = np.asscalar(match)  # convert 1x1 array to a numbers
        return match, match
    else:
        return [], []


def generate_anchor_positive(ids,batch_num=128):
    """
    :param ids: index array of the same identity
    :return: all pairs of [anchor, positive]. position matters
    """
    if len(ids) < 2 :
        #print("number of picture less than 2")
        return []
    results = []
    datasets = set()
    datasets.add(ids[0])
    datasets.add(ids[1])
    results.append(np.array([ids[0], ids[1]]))
    results.append(np.array([ids[1], ids[0]]))
    for idx in range(2, len(ids)):
        for elem in datasets:
            results.append(np.array([ids[idx], elem]))
            results.append(np.array([elem, ids[idx]]))
        datasets.add(ids[idx])
    results = np.array(results)
    if batch_num != None:
        if len(results) < batch_num:
            #copy itself to fill to the batch_num
            list = np.random.choice(len(results), batch_num-len(results))
            results = np.vstack( (results, results[list]))
        else:
            list = np.random.choice(len(results),batch_num)
            results = results[list]
    return results

def get_unique_labels(labels):
    """

    :param labels: unique labels preserving orders from the label array
    :return:
    """
    _, idx = np.unique(labels, axis=0, return_index=True)
    return labels[np.sort(idx)]


def get_triplets(labels, pairs):
    """ get a <anchor, positive, negative > without any mining.
    :param labels:  labels
    :param pairs:  [anchor, positive]
    :return:  triplets index for  [anchor, positive, negative]
    """
    triplets = []

    for idx in range(len(pairs)):
        # ng_ids, = np.where((labels != labels[pairs[idx, 0]]).any(axis=1))
        ng_ids, = np.where((labels != labels[pairs[idx, 0]]))
        ng_id = random.choice(ng_ids)
        triplets.append(np.array([pairs[idx, 0], pairs[idx, 1], ng_id]))
    return np.array(triplets)


def generate_triplet_data(codes,labels,batch_num = 128):
    """
    this function is used mostly for generating triplet data offline
    It consumes massive memory
    :param codes:
    :param labels:
    :return:
    """
    unique_labels = get_unique_labels(labels)
    data_anchor = np.zeros((1,7,7,2048))
    data_pos = np.zeros((1,7,7,2048))
    data_neg = np.zeros((1,7,7,2048))
    data_y = np.zeros((1),dtype='int32')
    pairs = []
    triplets = []
    yield_idx = 0
    batch_idx = yield_idx

    for ID in unique_labels:

        ids, = np.where((labels == ID))
        # get anchor-positive pairs.
        ap_pair = generate_anchor_positive(ids, batch_num)
        if len(pairs) > 0 and len(ap_pair) > 0:
            pairs = np.vstack((pairs, ap_pair))
        elif len(ap_pair) > 0:
            pairs = ap_pair

        batch_idx = len(pairs)

        while (yield_idx + batch_num <= batch_idx):
            triplets = get_triplets(labels, pairs[yield_idx:yield_idx + batch_num])
            if len(triplets) > 0:
                data_anchor = np.vstack((data_anchor,codes[triplets[:, 0]]))
                data_pos = np.vstack((data_pos, codes[triplets[:, 1]]))
                data_neg = np.vstack((data_neg, codes[triplets[:, 2]]))
                data_y = np.vstack((data_y,labels[triplets[:, 0]].reshape(-1,1)))
                yield_idx += batch_num
            else:
                logger.warn("No triplet found ")

    if yield_idx < batch_idx:
        triplets = get_triplets(labels, pairs[yield_idx:batch_idx])
        data_anchor = np.vstack((data_anchor, codes[triplets[:, 0]]))
        data_pos = np.vstack((data_pos, codes[triplets[:, 1]]))
        data_neg = np.vstack((data_neg, codes[triplets[:, 2]]))
        data_y = np.vstack((data_y,labels[triplets[:, 0]].reshape(-1,1)))
        yield_idx = batch_idx

    data_anchor = data_anchor[1:,:]
    data_pos = data_pos[1:,:]
    data_neg = data_neg[1:,:]
    data_y = data_y[1:]

    # shuffle data
    indics = np.arange(len(data_y))
    np.random.shuffle(indics)
    data_anchor = data_anchor[indics,:]
    data_pos = data_pos[indics,:]
    data_neg = data_neg[indics,:]
    data_y = data_y[indics,:]
    return  [data_anchor,data_pos,data_neg],data_y

class EmbeddingUpdator(Callback):
    "Keras Callback implemenation of saving the model and update embeddings"
    def __init__(self, filepath,  train_codes, train_labels, data_dir, monitor='val_loss',verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1 ):
        super(EmbeddingUpdator, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.train_codes = train_codes
        self.train_labels = train_labels
        self.data_dir = data_dir

        if mode not in ['auto', 'min', 'max']:
            logger.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logger.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)

                        # update the embedding
                        global embeddings
                        output = self.model.predict([self.train_codes,self.train_codes,self.train_codes])
                        embeddings = output[:,0:embedding_num]

                        file_name = os.path.join(self.data_dir, 'embeddings.npy')
                        np.save(open(file_name, 'wb'),embeddings)

                        file_name = os.path.join(self.data_dir, 'embedding_labels.npy')
                        np.save(open(file_name, 'wb'),self.train_labels)
                        print("Embeddings updated")

                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class tripletDataGenerator(object):
    "data generator for keras"
    def __init__(self,batch_size, classifier_filename,shuffle = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.classifier_filename = classifier_filename

    def generate(self, data, embedding_layer):
        "generate batch of sample"
        x_codes = data
        global embeddings
        global labels

        while True:
            if self.shuffle == True:
                indics = np.arange(len(labels))
                np.random.shuffle(indics)
                x_codes = x_codes[indics]
                labels = labels[indics]
                embeddings = embeddings[indics]

            #pairs = []
            #triplets = []
            yield_idx = 0
            #batch_idx = yield_idx
            # unique index needs to presever otherwise it will be always sorted from small to big
            #unique_labels = get_unique_labels(labels)
            batch_num = self.batch_size

            # google's facenet online triplet generator
            # use all anchor-positive pairs
            # semi hard negative that  ||anchor_embedding - positive_embedding|| < ||anchor_embedding - negative_embedding||
            # since the x_codes and labels are already shuffed and randomized, will just iterate through all ids
            step = 0
            while (step < len(labels)//self.batch_size):
                triplets = []
                batch_idics = np.arange(batch_num*step,batch_num*(step+1))
                batch_codes = x_codes[batch_idics]
                batch_labels = labels[batch_idics]
                batch_embeddings = embeddings[batch_idics]
                for y in batch_labels:
                    pos_ids = np.asarray(np.where((batch_labels == y)))[0]
                    if len(pos_ids) < 2:
                        continue
                    all_pos_embeddings = batch_embeddings[pos_ids[1:]]
                    sorted_dist_pos, sorted_pos_idx = l2_similarity(batch_embeddings[pos_ids[0]], all_pos_embeddings)
                    # the largest l2  <anchor, positive>  
                    if sorted_dist_pos[-1] > 0 :
                        max_pos_idx = pos_ids[sorted_pos_idx[-1]]
                    else:
                        continue  
                    ng_idcs, =np.where((batch_labels!=y))
                    if len(ng_idcs) < 1:
                        continue
                    all_ng_embeddings = batch_embeddings[ng_idcs]
                    sorted_dist_neg, sorted_neg_idx = l2_similarity(batch_embeddings[pos_ids[0]], all_ng_embeddings)
                    # the hardest <anchor, positive> and the hardest <anchor, negative>
                    min_neg_idx = ng_idcs[sorted_neg_idx[0]]
                    triplets.append(np.array([pos_ids[0]+batch_num*step , max_pos_idx + batch_num*step, min_neg_idx+batch_num*step]))
                triplets = np.array(triplets)
                yield [x_codes[triplets[:, 0]], x_codes[triplets[:, 1]], x_codes[triplets[:, 2]]], np.ones(len(triplets))
                step +=1


            # the last batch incomplete
            if len(labels) -batch_num*step> 0:
                triplets = []
                batch_idics = np.arange(batch_num*step,len(labels))
                batch_codes = x_codes[batch_idics]
                batch_labels = labels[batch_idics]
                batch_embeddings = embeddings[batch_idics]
                for y in batch_labels:
                    pos_ids = np.asarray(np.where((batch_labels == y)))[0]
                    if len(pos_ids) < 2:
                        continue
                    sorted_dist_pos, sorted_pos_idx = l2_similarity(batch_embeddings[pos_ids[0]],batch_embeddings[pos_ids])
                    # the largest l2  <anchor, positive>
                    if sorted_dist_pos[-1] > 0 :
                        max_pos_idx = pos_ids[sorted_pos_idx[-1]]
                    else:
                        continue

                    ng_idcs, = np.where((batch_labels != y))
                    if len(ng_idcs) < 1:
                        continue
                    all_ng_embeddings = batch_embeddings[ng_idcs]
                    sorted_dist_neg, sorted_neg_idx = l2_similarity(batch_embeddings[pos_ids[0]], all_ng_embeddings)
                    # #the hardest <anchor, positive> and the hardest <anchor, negative>
                    min_neg_idx = ng_idcs[sorted_neg_idx[0]]
                    triplets.append(np.array([pos_ids[0] + batch_num * step, max_pos_idx + batch_num * step,\
                                              min_neg_idx + batch_num * step]))
                triplets = np.array(triplets)
                yield [x_codes[triplets[:, 0]], x_codes[triplets[:, 1]], x_codes[triplets[:, 2]]], np.ones(len(triplets))



class tripletValidationDataGenerator(object):
    "data generator for keras"
    def __init__(self,batch_size, shuffle = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def generate(self, codes,labels):
        "generate batch of sample"
        #unique_labels = get_unique_labels(labels)
        while True:
            if self.shuffle == True:
                indics = np.arange(len(labels))
                np.random.shuffle(indics)
                codes = codes[indics]
                labels = labels[indics]

            step = 0
            while (step < len(labels)//self.batch_size):
                batch_idics = np.arange(self.batch_size*step,self.batch_size*(step+1))
                batch_codes = codes[batch_idics]
                batch_labels = labels[batch_idics]
                unique_batch_labels = get_unique_labels(batch_labels)
                for y in unique_batch_labels:                
                    pos_ids = np.asarray(np.where((batch_labels == y)))[0]
                    if len(pos_ids) < 2:
                        continue
                    ng_idcs, =np.where((batch_labels!=y))
                    if len(ng_idcs) < 1:
                        continue    

                    ap_pair = generate_anchor_positive(pos_ids, batch_num=len(ng_idcs))
                    yield [batch_codes[ap_pair[:, 0]], batch_codes[ap_pair[:, 1]], batch_codes[ng_idcs]], np.ones(len(ap_pair))
                step +=1

            # the last batch incomplete

            if len(labels) - self.batch_size*step> 0:
                batch_idics = np.arange(self.batch_size*step,len(labels))
                batch_codes = codes[batch_idics]
                batch_labels = labels[batch_idics]
                unique_batch_labels = get_unique_labels(batch_labels)
                for y in unique_batch_labels:
                    pos_ids = np.asarray(np.where((batch_labels == y)))[0]
                    #print("y: ",y, pos_ids)
                    if len(pos_ids) < 2:
                        continue
                    ng_idcs, =np.where((batch_labels!=y))
                    if len(ng_idcs) < 1:
                        continue    

                    ap_pair = generate_anchor_positive(pos_ids, batch_num=len(ng_idcs) )

                    #print("anchor\n",batch_labels[ap_pair[:, 0]],"\npositive\n",batch_labels[ap_pair[:, 1]],"\nnegative\n",batch_labels[ng_choice])
                    yield [batch_codes[ap_pair[:, 0]], batch_codes[ap_pair[:, 1]], batch_codes[ng_idcs]], np.ones(len(ap_pair))     
                


class TripletClassifier():
    def __init__(self, classfier_filename, data_dir, num_embeddings):
        self.classifier_filename = classfier_filename
        self.num_embeddings = num_embeddings
        self.alpha = 0.2
        self.thres = 1.0
        self.data_dir = data_dir
        #K.set_image_data_format('channels_last')


    def prepare(self):
        self.train_codes = np.load(os.path.join(self.data_dir, 'train_codes.npy'))
        self.test_codes = np.load(os.path.join(self.data_dir, 'test_codes.npy'))
        self.train_labels = np.squeeze(np.load(os.path.join(self.data_dir, 'train_labels.npy')))
        self.test_labels = np.squeeze(np.load(os.path.join(self.data_dir, 'test_labels.npy')))
        self.test_idcs = np.load(os.path.join(self.data_dir,'test_labels_idcs.npy'))
        self.random_draw_codes = np.load(os.path.join(self.data_dir, 'random_draw_codes.npy'))
        self.random_draw_label = np.squeeze(np.load(os.path.join(self.data_dir, 'random_draw_labels.npy')))

        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_idcs, val_idcs = next(ss.split(self.train_codes, self.train_labels))
        self.x_codes = self.train_codes[train_idcs]
        self.y_labels = self.train_labels[train_idcs]
        self.val_codes = self.train_codes[val_idcs]
        self.val_labels = self.train_labels[val_idcs]


        # classifier
        logger.info("train codes shape {}".format(self.x_codes.shape))
        logger.info("validation codes shape {}".format(self.val_codes.shape))
        logger.info("train label shape {}".format(self.y_labels.shape))
        logger.info("validation label shape {}".format(self.val_labels.shape))
        logger.info("test codes  shape {}".format(self.test_codes.shape))
        logger.info("test labels  shape {}".format(self.test_labels.shape))

        input = Input(shape=(7, 7, 2048))
        x = Flatten(name='flatten')(input)
        x = Dropout(0.5)(x)
        x = Dense(self.num_embeddings, kernel_initializer= TruncatedNormal(stddev=0.05),
                 use_bias=True, bias_initializer='zeros',
                name='embeddings', kernel_regularizer=l2(0.001))(x)
        x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)

        branch = Model(inputs=input,outputs=x)

        anchors = Input(shape=(7, 7, 2048))
        positives = Input(shape=(7, 7, 2048))
        negatives = Input(shape=(7, 7, 2048))

        anchors_embeddings = branch(anchors)
        positives_embeddings = branch(positives)
        negatives_embeddings = branch(negatives)
        merged = concatenate([anchors_embeddings, positives_embeddings,negatives_embeddings],axis=1)

        self.model = Model([anchors, positives,negatives], merged)
        self.model.compile(loss= triplet_loss,
                           optimizer = Adam(lr=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                            metrics=[accuracy])
        self.model.summary()

    def train(self):

        ## split training set further for training and validation set

        start_time = time.time()
        # used for choosing triplets to train
        global embeddings
        pca = PCA(n_components=embedding_num)
        temp = np.array(self.x_codes).reshape((self.x_codes.shape[0],-1))
        embeddings = pca.fit_transform(temp)

        global labels
        labels = self.y_labels

        logger.info('Training starts...' )
        checkpointer = EmbeddingUpdator(filepath=self.classifier_filename,train_codes=self.x_codes, train_labels = self.y_labels , data_dir = self.data_dir,
                monitor ='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',patience=20)
        tbCallBack = TensorBoard(log_dir='./logs',batch_size= batch_num, histogram_freq=0, write_graph=True,
                                                 write_images=True, write_grads = True)
        tbCallBack.set_model(self.model)
        lrate = LearningRateScheduler(step_decay)

        train_data_params = {
            'batch_size':batch_num,
            'shuffle':True,
            'classifier_filename': self.classifier_filename
        }
        val_batch_size = 64
        val_data_params = {
            'batch_size':val_batch_size,
            'shuffle':True
        }
        train_data_generator = tripletDataGenerator(**train_data_params).generate(self.x_codes,self.model)
        val_data_generator = tripletValidationDataGenerator (**val_data_params).generate(self.val_codes,self.val_labels)

        evaluation = self.model.fit_generator(train_data_generator,
                        steps_per_epoch=(len(self.y_labels)+ batch_num-1) // batch_num,
                        epochs=1000,
                        verbose=1,
                        callbacks=[checkpointer,early_stopping,tbCallBack,lrate],
                        validation_data=val_data_generator,
                        validation_steps = (val_batch_size * len(np.unique(self.val_labels))+ batch_num-1)//batch_num,
                        use_multiprocessing = False)

        logger.info('Completed in {} seconds'.format(time.time() - start_time))
        logger.info("Training result {}".format(evaluation))
        pickle.dump(evaluation.history, open("../model/cnn_train_history.pickle", "wb"))

        return 0

    def train_history_visual(self):

        # summarize history for accuracy
        with open("../model/cnn_train_history.pickle", 'rb') as pickle_file:
            history = pickle.load(pickle_file)
        plt.figure(figsize=(6,3))

        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='lower right')
        axes = plt.gca()
        axes.set_ylim([0.0, 1.05])

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')

        plt.show()

    def predict(self, test_codes, anchor_embeddings,anchor_labels, anchor_thres,top_k = 5):
        test_embeddings = self.model.predict([test_codes,test_codes,test_codes],batch_size = 128)
        test_embeddings = test_embeddings[:,0:embedding_num]
        match_labels_thres = []
        # for query_embedding in test_embeddings:
        #     dist = l2_distance(query_embedding[None, :], anchor_embeddings)
        #     sorted_idcs = np.array(sorted(range(len(dist)), key=lambda k: dist[k]))
        #     top_k_idcs = sorted_idcs[:top_k]
        #     top_k_match = []
        #     # print("top_k_labels:",np.array(anchor_labels)[top_k_idcs])
        #     # print("top_k_dist",dist[top_k_idcs])
        #     # print("thres: ",np.array(anchor_thres)[top_k_idcs])
        #     for cand in top_k_idcs:
        #         if dist[cand] < anchor_thres[cand]:
        #             top_k_match.append(anchor_labels[cand])
        #     #get the majority voted labels in top_k possible matching labels
        #     if len(top_k_match )>0:
        #         unique_labels,first_idcs, counts = np.unique(np.array(top_k_match), return_counts=True,return_index=True)
        #         # print("label candidate:",unique_labels)
        #         # print("label size", len(unique_labels))
        #         if len(unique_labels) >0 :
        #             match_idx = np.where(counts==counts.max())
        #             if  np.asarray(match_idx).size >=2:  #tied match
        #                 # the first occuring one in the top_k . It has the smallest distance
        #                 match_idx = np.argsort(first_idcs[match_idx])
        #                 match = unique_labels[match_idx[0]]
        #                 match = np.ravel(match)
        #             else:
        #                 match = unique_labels[match_idx]
        #             match_labels_thres.append(match)
        #     else:
        #         match_labels_thres.append([])


        # distance matching works better in small set
        for query_embedding  in test_embeddings:
            #match_thres, match = query_top_match(query_embedding,anchor_embeddings,np.array(anchor_labels),self.thres,top_k=5)
            match_thres, match = query_top_match(query_embedding,anchor_embeddings,anchor_labels,anchor_thres,top_k=1)
            match_labels_thres.append(match_thres)

        return match_labels_thres

    def generate_labels_embeddings_threshold_2(self):
        self.model.load_weights(self.classifier_filename)
        #embeddings = np.load(os.path.join(self.data_dir, 'embeddings.npy'))
        #y_labels = np.load(os.path.join(self.data_dir, 'embedding_labels.npy'))
        embeddings = self.model.predict([self.x_codes, self.x_codes, self.x_codes],batch_size = 128)
        y_labels = self.y_labels
        embeddings = embeddings[:,0:embedding_num]
        label_embeddings = []
        label_thres = []
        labels = []
        neg_dist = []
        pos_dist = []
        for id in np.unique(y_labels):
            same_labels_ids = np.where(y_labels == id )[0]
            ng_labels_ids = np.where(y_labels !=id )[0]
            label_embeddings.append(embeddings[same_labels_ids].mean(axis=0))

            for i in same_labels_ids:
                for j in same_labels_ids:
                    #neg_dist.append(l2_distance(embeddings[j][None,:], embeddings[i][None,:]))
                    pos_dist.append(l2_distance(embeddings[j][None,:], embeddings[i][None,:]))
            # dist_pos = l2_distance(embeddings[same_labels_ids[1:]],embeddings[same_labels_ids[0]])
            # dist_neg = l2_distance(embeddings[ng_labels_ids], embeddings[same_labels_ids[0]])
            # positive = np.array(dist_pos,dtype='double')
            # negative = np.array(dist_neg,dtype='double')
            # thres = bob.measure.eer_threshold(negative,positive)
            #label_thres.append(np.min(negative))
            label_thres.append(np.mean(pos_dist)  )
            labels.append(id)
            #label_thres.append(np.mean(positive) - 0.4 )
        #label_embeddings = np.vstack(label_embeddings)
        #return label_embeddings,np.array(labels),np.mean(label_thres)
        return embeddings,y_labels,np.mean(label_thres)


    def evaluate(self):
        true_accept = 0.0
        false_accept = 0.0
        true_reject = 0.0
        false_reject = 0.0

        anchor_embeddings, anchor_labels, anchor_thres = self.generate_labels_embeddings_threshold_2()
        # self.thres = np.mean(anchor_thres) 
        logger.info("Distance threshold {0:.4f}:".format(anchor_thres) )

        #evaluate test accuracy 
        # [anchor, pos, neg], y = generate_triplet_data(self.test_codes, self.test_labels, 16)
        # logger.info("Total testing triplet number: {}".format(len(y)))
        # test_embeddings = self.model.predict([anchor, pos, neg], batch_size=128)

        # logger.info("Testing model accuracy...")
        # with tf.Session() as test:
        #     acc_score = accuracy(y,test_embeddings)
        #     logger.info("accuracy rate: {0:.4f}".format(acc_score.eval(session = test)))

        # logger.info("Testing model VAL/FA...")
        # test_embeddings = test_embeddings[:,0:embedding_num]
        # grand_same = 0
        # grand_diff = 0
        # predict_diff = 0
        # grand_same_predict_same = 0
        # grand_same_predict_diff = 0
        # grand_diff_predict_same = 0

        # for  (id1, id2) in itertools.combinations(test_embeddings,2):
        #     #print(np.where(test_embeddings == id1))
        #     id1_label = y[np.where(test_embeddings == id1)[0][0]]
        #     id2_label = y[np.where(test_embeddings == id2)[0][0]]
        #     if id1_label == id2_label:
        #         grand_same += 1
        #     else:
        #         grand_diff += 1    
        #     if l2_distance(id1[None,:], id2[None,:]) < self.thres:
        #         if id1_label == id2_label:
        #             grand_same_predict_same += 1
        #         else:
        #             grand_diff_predict_same += 1
        #     else:
        #         predict_diff += 1
        #         if id1_label == id2_label:
        #             grand_same_predict_diff += 1
        # logger.info("VAL rate: {0:.4f}".format(grand_same_predict_same/grand_same))
        # logger.info("FA rate: {0:.4f}".format(grand_diff_predict_same/grand_diff))
        # logger.info("FR rate: {0:.4f}".format(grand_same_predict_diff/grand_same))

        #anchor_embeddings, anchor_thres, anchor_labels = self.generate_labels_embeddings_threshold()
        # for idx in range(len(anchor_labels)):
        #     logger.info("label {} threshold: {}".format(anchor_labels[idx], anchor_thres[idx]))

        logger.info("Testing all authorized labels...")
        predictions_thres = self.predict(self.test_codes,anchor_embeddings,anchor_labels, anchor_thres,top_k=1)

        all_test = 0.0
        for test_id in range(len(self.test_labels)):
            if np.asarray(np.where(predictions_thres[test_id] == self.test_labels[test_id])).size==0:
                logger.info("data id  {} false rejected as label {}, true label {}"
                            .format(self.test_idcs[test_id],predictions_thres[test_id],self.test_labels[test_id]))
                false_reject += 1.0

            else:
                true_accept += 1.0
            all_test += 1.0

        TR = true_accept/all_test
        FR = false_reject/all_test

        logger.info("true_acceptance: {0:.4f}".format(TR))
        logger.info("false_rejection: {0:.4f}".format(FR))

        logger.info("Testing random draw labels...")
        # label_seen = 0
        # for in_label in np.unique(self.val_labels):
        #     tmp = len(np.where(self.random_draw_label == in_label)[0])
        #     if tmp > 0:
        #         label_seen +=tmp
        # logger.info("{} out of {} labels are authorized.".format(label_seen, self.random_draw_label.size))

        # some are in the system some are not authorized
        predictions_thres = self.predict(self.random_draw_codes,anchor_embeddings,anchor_labels, anchor_thres,top_k=1)
        # what went wrong
        true_accept = 0.0
        false_accept = 0.0
        true_reject = 0.0
        false_reject = 0.0
        all_test = 0.0
        for test_id in range(len(self.random_draw_codes)):
            all_test += 1 
            if (self.random_draw_label[test_id]== self.train_labels).any() == True:
                # in the system
                if np.asarray(predictions_thres[test_id]).size == 0 or predictions_thres[test_id] != self.random_draw_label[test_id]:
                    false_reject += 1.0
                    logger.info("data id  {} false rejected as label {}, true label {}"
                                .format(test_id, predictions_thres[test_id], self.random_draw_label[test_id]))
                else:
                    true_accept += 1.0
                    logger.info("data id  {} accepted label  {} true label {} "
                                .format(test_id, predictions_thres[test_id],self.random_draw_label[test_id] ))
            else:
                # not in the system
                if np.asarray(predictions_thres[test_id]).size > 0:
                    false_accept += 1.0
                    logger.info("data id  {} false accepted as label {}, none autherized label {}"
                                .format(test_id, predictions_thres[test_id], self.random_draw_label[test_id]))
                else:
                    true_reject += 1.0
                    logger.info("data id  {} rejected label {}, none autherized"
                                .format(test_id, self.random_draw_label[test_id]))

        # logger.info("false_accept: {0:.4f}".format(false_accept))
        # logger.info("true_accept: {0:.4f}".format(true_accept))
        # logger.info("false_reject: {0:.4f}".format(false_reject))
        # logger.info("true_reject: {0:.4f}".format(true_reject))


        FAR = false_accept / all_test
        TR = true_reject/all_test
        logger.info("true_rejection: {0:.4f}".format(TR))
        logger.info("false_acceptance: {0:.4f}".format(FAR))

        # combine test_label and random draw
        self.test_labels = np.vstack((self.test_labels[:,None], self.random_draw_label[:,None]))
        self.test_codes = np.vstack((self.test_codes, self.random_draw_codes))

        # some are in the system some are not authorized
        predictions_thres = self.predict(self.test_codes,anchor_embeddings,anchor_labels, anchor_thres,top_k=1)
        # what went wrong
        true_accept = 0.0
        false_accept = 0.0
        true_reject = 0.0
        false_reject = 0.0

        for test_id in range(len(self.test_labels)):
            if (self.test_labels[test_id]== self.train_labels).any() == True:
                # in the system
                if np.asarray(predictions_thres[test_id]).size == 0 or predictions_thres[test_id] != self.test_labels[test_id]:
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


    clr = TripletClassifier(args.file_name,data_dir=args.input_dir,num_embeddings = embedding_num)
    clr.prepare()
    
    clr.train()
    clr.evaluate()
    clr.train_history_visual()

