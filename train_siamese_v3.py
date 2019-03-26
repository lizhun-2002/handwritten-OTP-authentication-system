# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:02:34 2017

@author: LZ

Trains a Siamese deep CNN on pairs of digits from the SDB19 dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
"""

import os
import os.path

import numpy as np
from PIL import Image

import random
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop,SGD,Adadelta
from keras import backend as K

from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


datapath='./data'
#datapath='E:/NIST Special Database 19'

epochs = 1000

# Helper: Save the model.
checkpointer = ModelCheckpoint(
#    filepath='./data/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    filepath=datapath+'/checkpoints/' + 'siamese' + '-' + 'SDB_19' + '_100epoch.hdf5',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=2)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=datapath+'/logs/')

for folder in ['checkpoints','logs']:
    if not os.path.exists(datapath + '/' + folder):
        os.makedirs(datapath + '/' + folder)
        
#==============================================================================
#  Load all images in specified folder to memory
#==============================================================================
# Should be: path/subfolders/images
# The images are sorted by name
def load_all_images(path):
    X, y = [], []
    i = 0
    for dirs in sorted(os.listdir(path)):
        subpath = os.path.join(path,dirs)
        # go on only when this is a folder
#        if not(os.path.isdir(subpath)):
        if not(os.path.isdir(subpath)) or dirs!='hsf_2':
            continue
        for filename in sorted(os.listdir(subpath)):
            imgpath = os.path.join(subpath,filename)
            img = Image.open(imgpath)
            img =img.convert('L')
            #img =img.convert('L').resize((28,28))
            width,hight=img.size
            img = np.asarray(img,dtype='float32')/255.
            X.append(img)
            #y.append(int(dirs))
            y.append(i)
        i+=1
    return np.array(X), np.array(y)
        
#==============================================================================
# Load writer information
#==============================================================================
# Should be: path/.mix files
# .mix files are sorted by name, the contents are alse sorted
# Todo: load images according to .mix file
def load_writer_info(path):
    y = []
    for dirs in sorted(os.listdir(path)):
        subpath = os.path.join(path,dirs)
        # go on only when this is a folder
#        if os.path.isdir(subpath):
        if os.path.isdir(subpath) or dirs!='hsf_2.mit':
            continue
        with open(subpath, 'r') as fin:
            while 1:
                line = fin.readline()
                if not line:
                    break
                parts=line.split(' ')
                if len(parts)==2:
                    writerID=int(parts[1][1:5])
                    y.append(writerID)
    return np.array(y)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    num_classes = len(digit_indices)
    #n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        n =len(digit_indices[d]) - 1
        if n < 1:
            continue
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            j = random.randrange(0, len(digit_indices[dn]))
            z1, z2 = digit_indices[d][i], digit_indices[dn][j]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(128, activation='relu'))
    return model


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def plot_eer_ndigits(y_true, y_pred, n=1, saveas='FRR_FAR_'):
    # plot frr and far for n digits. y_true is positive-negative label, y_pred is distance
    frrs, fars = [], []
    thres=np.arange(0,1,0.01)
    for thre in thres:
        pred = y_pred.ravel() < thre
        frr = 1 - np.mean(y_true[y_true==1] == pred[y_true==1])**n
        frrs.append(frr)
        far = 1 - np.mean(y_true[y_true==0] == pred[y_true==0])
        far = far**n
        fars.append(far)
    # compute intersection point
    ind = np.argmin(abs(np.array(frrs)-np.array(fars)))
    plt.figure()
    plt.xlabel('Similarity')  
    plt.ylabel('Accuracy')
    plt.plot(thres,frrs, label='FRR')
    plt.plot(thres,fars, label='FAR')
    # plot intersection point
    label_point = '('+'%.2f' % thres[ind]+','+'%.2f' % frrs[ind]+')'
    plt.plot(thres[ind],frrs[ind],"o")
    plt.text(thres[ind],frrs[ind]+0.1,label_point,fontsize=15,verticalalignment="bottom",horizontalalignment="left")
    plt.annotate('',xy=(thres[ind],frrs[ind]),xytext=(thres[ind],frrs[ind]+0.1),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    plt.legend()
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig(saveas+str(n), dpi=1000)
    plt.show()

def plot_roc_ndigits(y_true, y_pred, n, saveas='ROC_'):
    # plot FPR and TPR for n digits. y_true is positive-negative label, y_pred is distance
    fprs, tprs = [], []
    thres=np.arange(0,1,0.01)
    for thre in thres:
        pred = y_pred.ravel() < thre
        tpr = np.mean(y_true[y_true==1] == pred[y_true==1])**n
        tprs.append(tpr)
        fpr = 1 - np.mean(y_true[y_true==0] == pred[y_true==0])
        fpr = fpr**n
        fprs.append(fpr)
    plt.figure()
    plt.xlabel('False Positive Rate(FPR)')  
    plt.ylabel('True Positive Rate(TPR)')
    plt.plot(fprs,tprs)
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig(saveas+str(n), dpi=1000)
    plt.show()

def main():
    # Data should be placed like: train_siamese/30/hsf_0
    # We use only 500 writer data, 320 for training, 80 for validation, 100 for test
    print("Loading data from %s." % datapath)
    tr_pairs, tr_y, va_pairs, va_y, te_pairs, te_y = [],[],[],[],[],[]
    for dirs in sorted(os.listdir(datapath+'/train_siamese_all')):
        subpath = os.path.join(datapath+'/train_siamese_all',dirs)
        print("Loading data from %s." % subpath)
        X,d=load_all_images(subpath)
        y=load_writer_info(subpath)
        print('Length check:',len(d),len(y))
        assert len(d)==len(y)
        # Transform writer ID to [0,499]
        y=y-min(y)
    
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        x_train=X[np.where(y < 320)[0]]
        y_train=y[np.where(y < 320)[0]]
        x_val=X[np.where((y>=320) * (y<400))[0]]
        y_val=y[np.where((y>=320) * (y<400))[0]]
        x_test=X[np.where((y>=400) * (y<500))[0]]
        y_test=y[np.where((y>=400) * (y<500))[0]]
        
#        x_train=X[np.where(y < 2100)[0]]#hsf_0 to 3
#        y_train=y[np.where(y < 2100)[0]]
#        x_val=X[np.where((y>=3100) * (y<3600))[0]]#hsf_6
#        y_val=y[np.where((y>=3100) * (y<3600))[0]]
#        x_test=X[np.where((y>=3600) * (y<4100))[0]]#hsf_7
#        y_test=y[np.where((y>=3600) * (y<4100))[0]]
        
        input_shape = x_train.shape[1:]
        
        # create training+test positive and negative pairs
        digit_indices = [np.where(y_train == i)[0] for i in set(y_train)]
        tr_pairs0, tr_y0 = create_pairs(x_train, digit_indices)
        tr_pairs.extend(tr_pairs0)
        tr_y.extend(tr_y0)
        
        digit_indices = [np.where(y_val == i)[0] for i in set(y_val)]
        va_pairs0, va_y0 = create_pairs(x_val, digit_indices)
        va_pairs.extend(va_pairs0)
        va_y.extend(va_y0)
        
        digit_indices = [np.where(y_test == i)[0] for i in set(y_test)]
        te_pairs0, te_y0 = create_pairs(x_test, digit_indices)
        te_pairs.extend(te_pairs0)
        te_y.extend(te_y0)
        
    tr_pairs=np.array(tr_pairs)
    tr_y=np.array(tr_y)
    va_pairs=np.array(va_pairs)
    va_y=np.array(va_y)
    te_pairs=np.array(te_pairs)
    te_y=np.array(te_y)
    print('Train: ',tr_pairs.shape,'  Val: ',va_pairs.shape,'  Test: ',te_pairs.shape)

#    # network definition
#    base_network = create_base_network(input_shape)
#    
#    input_a = Input(shape=input_shape)
#    input_b = Input(shape=input_shape)
#    
#    # because we re-use the same instance `base_network`,
#    # the weights of the network
#    # will be shared across the two branches
#    processed_a = base_network(input_a)
#    processed_b = base_network(input_b)
#    
#    distance = Lambda(euclidean_distance,
#                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
#    
#    model = Model([input_a, input_b], distance)
#    
#    # train
#    rms = RMSprop()
#    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    
    print('Loading model.')
    saved_model=datapath+'/checkpoints/' + 'siamese' + '-' + 'SDB_19' + '_100epoch.hdf5'
    model = load_model(saved_model, custom_objects={'contrastive_loss': contrastive_loss,'accuracy': accuracy})
    model.compile(loss=contrastive_loss, optimizer=SGD(lr=0.0001, momentum=0.9), metrics=[accuracy])
    print('Model is loaded.')
    
#    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#              batch_size=128,
#              epochs=epochs,
#              validation_data=([va_pairs[:, 0], va_pairs[:, 1]], va_y),
##              steps_per_epoch=1000,
##              validation_steps=100,
#              callbacks=[checkpointer, early_stopper, tensorboard])
    
    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]],verbose=1)
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]],verbose=1)
    te_acc = compute_accuracy(te_y, y_pred)
    
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    # FRR: false rejection rate
    frr = 1 - compute_accuracy(te_y[te_y==1], y_pred[te_y==1])
    print('* FRR on test set: %0.2f%%' % (100 * frr))
    # FAR: false acceptance rate
    far = 1 - compute_accuracy(te_y[te_y==0], y_pred[te_y==0])
    print('* FAR on test set: %0.2f%%' % (100 * far))
    
    plot_eer_ndigits(te_y, y_pred, 1)
    plot_eer_ndigits(te_y, y_pred, 4)
    plot_eer_ndigits(te_y, y_pred, 6)
    plot_eer_ndigits(te_y, y_pred, 8)
    
    plot_roc_ndigits(te_y, y_pred, 1)
    plot_roc_ndigits(te_y, y_pred, 4)
    plot_roc_ndigits(te_y, y_pred, 6)
    plot_roc_ndigits(te_y, y_pred, 8)

if __name__ == '__main__':
    main()