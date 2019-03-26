# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 00:56:47 2017

@author: LZ
"""
import os
import os.path

import numpy as np
from PIL import Image

from keras.models import load_model

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt



datapath='./data'
#datapath='E:/NIST Special Database 19'


#==============================================================================
#  Load all images in specified folder to memory
#==============================================================================
# Should be: path/subfolders/images
# The images are sorted by name
def load_all_images_3c(path):
    X, y = [], []
    i = 0
    for dirs in sorted(os.listdir(path)):
        subpath = os.path.join(path,dirs)
        # go on only when this is a folder
        if not(os.path.isdir(subpath)):
            continue
#        if len(os.listdir(subpath))<80:
#            continue
        for filename in sorted(os.listdir(subpath)):
            imgpath = os.path.join(subpath,filename)
            img = Image.open(imgpath)
#            img =img.convert('L')
            #img =img.convert('L').resize((28,28))
            width,hight=img.size
            img = np.asarray(img,dtype='float32')/255.
            X.append(img)
            #y.append(int(dirs))
            y.append(i)
        i+=1
    return np.array(X), np.array(y)


def onehot2simple(y_onehot):
#    y_sim = np.array([list(onehot).index(1) for onehot in y_onehot])
    y_sim = np.array([np.argmax(onehot) for onehot in y_onehot])
    return y_sim


def plot_digit_example(digits, labels, saveas='number', num_r=3, num_c=3):
    for i in range(num_r*num_c):
        ax=plt.subplot(num_r,num_c,i+1)
        ax.set_title(str(labels[i]))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(digits[i], cmap=plt.get_cmap('gray'))
    # show the plot
    #plt.tight_layout() # leads to very very small digit figures for linux; work well for win
    plt.subplots_adjust(hspace=0.5,wspace=0)
    foo_fig = plt.gcf() # 'get current figure'
    foo_fig.savefig(saveas, dpi=1000, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',saveas='cm', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)

    plt.figure()    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    foo_fig = plt.gcf() # 'get current figure'
#    foo_fig.savefig('confusion_matrix.eps', format='eps', dpi=1000) 
    foo_fig.savefig(saveas, dpi=1000, bbox_inches='tight')
    plt.show()


def main():
    print('Loading data.')
    X_test, y_test = load_all_images_3c(datapath+'/test')
    print('Loading model.')
    saved_model=datapath+'/checkpoints/' + 'cnn' + '-' + 'SDB_19' + '_100epoch.hdf5'
    model_cnn = load_model(saved_model, compile=False)
    print('Model is loaded.')
    # predict classification probability
    y_d_proba=model_cnn.predict(
            X_test,
            batch_size=128,
            verbose=1)
    # convert proba to classification results
    y_d_pred = onehot2simple(y_d_proba)

    print('* Accuracy on test set: %0.2f%%' % (100 * np.mean(y_d_pred == y_test)))


    # confusion_matrix
    results = confusion_matrix(y_test, y_d_pred, np.arange(len(set(y_test))))#labels=[0,1,2])

    print('Confusion_matrix on test data is:')
    print(results)

    plot_confusion_matrix(
            results, 
            classes=np.arange(len(set(y_test))),#data.classes, 
            normalize=True, 
            title='Confusion matrix',
            saveas='Confusion_matrix_normalized')
    plot_confusion_matrix(
            results, 
            classes=np.arange(len(set(y_test))),#data.classes, 
            normalize=False, 
            title='Confusion matrix',
            saveas='Confusion_matrix_not_normalized')

    false_example_index=np.array(np.where(y_d_pred != y_test)[0])
#    false_example=X_test(y_d_pred != y_test)
    print(false_example_index.shape)
    for i in range(10):
        rn=np.random.choice(false_example_index,9,replace=False)
        plot_digit_example(X_test[rn],y_d_pred[rn],'false_prediction'+str(i),3,3)


if __name__ == '__main__':
    main()