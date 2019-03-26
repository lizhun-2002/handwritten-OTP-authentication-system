# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:23:54 2017

@author: LZ
"""

import os
import os.path

import numpy as np
from PIL import Image

from keras.models import load_model
from keras import backend as K
from keras.backend import clear_session

import mysql.connector
import base64
import io

import datetime
import random
import time

from scipy import misc

#==============================================================================
#  Load all images in specified folder to memory
#==============================================================================
# Should be: path(user)/subfolders(digit)/images
# The images are sorted by name
def load_all_images(path):
    X, y = [], []
    i = 0
    for dirs in sorted(os.listdir(path)):
        subpath = os.path.join(path,dirs)
        # go on only when this is a folder
        if not(os.path.isdir(subpath)):
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

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def load_img_from_db(userID):
    config = {
      'user': 'root',
      'password': 'daewoo12',
      'host': '18.220.129.26',
      'database': 'Login',
      'raise_on_warnings': True,
    }
    cnx = mysql.connector.connect(**config)
    #cnx = mysql.connector.connect(user='scott', database='employees')
    cursor = cnx.cursor()
    query = ("SELECT IMG1, IMG2, IMG3, IMG4, RNG FROM Register_info WHERE ID= %s")
    cursor.execute(query,(userID,))
    # get tuple ('IMG1', 'IMG2', 'IMG3', 'IMG4', 'RNG')
    res = cursor.fetchall()[0] #fetchall() return a tuple of tuple, 
    images=[]
    for img in res[:-1]:
        _bytes=base64.b64decode(img)
        _image = Image.open(io.BytesIO(_bytes))
        #_image = _image.resize((128,128))
        _image = 1-np.asarray(_image,dtype='float32')/255.
        images.append(_image[:,:,3])
    #print(cursor.column_names)
    cursor.close()
    cnx.close()
    return np.array(images), res[-1]
    
def update_res_to_db(res):
    #res is a list, contains the value(Pred_num, Is_user, ID) to be uploaded. E.g. ['5555', '0', 'YJ1']
    config = {
      'user': 'root',
      'password': 'daewoo12',
      'host': '18.220.129.26',
      'database': 'Login',
      'raise_on_warnings': True,
    }
    cnx = mysql.connector.connect(**config)
    #cnx = mysql.connector.connect(user='scott', database='employees')
    cursor = cnx.cursor()
    #cursor.execute('insert into Register_info (Pred_num, Is_user) values (%s, %s)', res)
    cursor.execute('UPDATE Register_info SET Pred_num=%s, Is_user=%s WHERE ID= %s', res)
    flag=cursor.rowcount # sucess 1; fail 0
    # 提交事务
    cnx.commit()
    cursor.close()
    cnx.close()
    return flag


def download_img_from_db(userID, savepath):
    # load img from database, and save img to pc
    config = {
      'user': 'root',
      'password': 'daewoo12',
      'host': '18.220.129.26',
      'database': 'Login',
      'raise_on_warnings': True,
    }
    cnx = mysql.connector.connect(**config)
    #cnx = mysql.connector.connect(user='scott', database='employees')
    cursor = cnx.cursor()
    query = ("SELECT IMG1, IMG2, IMG3, IMG4, RNG FROM Register_info WHERE ID= %s")
    cursor.execute(query,(userID,))
    # get tuple ('IMG1', 'IMG2', 'IMG3', 'IMG4', 'RNG')
    res = cursor.fetchall()[0] #fetchall() return a tuple of tuple, 
    for img in res[:1]:
        userID='qwer'
        img=res[0]
        _bytes=base64.b64decode(img)
        _image = Image.open(io.BytesIO(_bytes))
#        plt.imshow(_image, cmap=plt.get_cmap('gray'))
        _image = 255-np.asarray(_image,dtype='float32')
        _image = _image[:,:,3]
        _image = misc.toimage(_image, cmin=0, cmax=255)
        _image.save(os.path.join(savepath,gen_file_name()+'.png'))
    cursor.close()
    cnx.close()

#def save_img_to_pc(path, _bytes):
#    fh = open(os.path.join(path,gen_file_name()), "wb")
#    fh.write(_bytes)
#    fh.close()
    
def gen_file_name():
    nowTime=datetime.datetime.now().strftime("%Y%m%d%H%M%S")#生成当前时间  
    randomNum=random.randint(0,100)#生成的随机整数n，其中0<=n<=100  
    if randomNum<=10:  
        randomNum=str(0)+str(randomNum)  
    uniqueNum=str(nowTime)+str(randomNum)    
    return uniqueNum




#datapath='./data'
datapath='E:/NIST Special Database 19'
subpath1 = datapath+'/run_siamese'+'/user'
subpath2 = datapath+'/run_siamese'+'/input'

#==============================================================================
#download_img_from_db('pp', subpath2)
#==============================================================================

#for i in range(10):
#    os.mkdir(subpath1+'/'+str(i))

# Load CNN model
saved_model=datapath+'/checkpoints/' + 'cnn' + '-' + 'SDB_19' + '_100epoch.hdf5'
model_cnn = load_model(saved_model)
# Load Siamese model
saved_model=datapath+'/checkpoints/' + 'siamese' + '-' + 'SDB_19' + '_100epoch.hdf5'
model_siamese = load_model(saved_model, custom_objects={'contrastive_loss': contrastive_loss})

# Load user data
X,d=load_all_images(subpath1)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

noise=0
while 1:
    # Load input data
    X_input, rn=load_img_from_db('pp')
#    X_input,d_input=load_all_images(subpath2)
    X_input = X_input.reshape(X_input.shape[0], X_input.shape[1], X_input.shape[2], 1)
    
    ########################## Handwriting recognition ############################
    # prepare data: need 3 channels
    X_input_3c=np.array([np.dstack((img,)*3) for img in X_input])
    
    # predict classification probability
    y_d_proba=model_cnn.predict(
            X_input_3c,
            batch_size=32,
            verbose=1)
    # convert proba to classification results
    y_d_pred = np.array([list(onehot).index(max(onehot)) for onehot in y_d_proba])
    
    ########################## Witer verification #################################
    digit_indices = [np.where(d == i)[0] for i in set(y_d_pred)]
    #digit_indices = [np.where(d == i)[0] for i in range(10)]
    
    te_pairs=[]
    num_of_input=len(set(y_d_pred)) # num of input
    for i in range(num_of_input):
        for dig in digit_indices[i]:
            te_pairs += [[X[dig], X_input[i]]]
    te_pairs=np.array(te_pairs)
    
    
    # compute final accuracy on training and test sets
    y_pred = model_siamese.predict([te_pairs[:, 0], te_pairs[:, 1]],verbose=1)
    te_acc = compute_accuracy(np.ones(len(y_pred)), y_pred)
    
    
    pred_num=0
    for k in range(len(y_d_pred)):
        pred_num+=y_d_pred[k]*10**(len(y_d_pred)-1-k)
    
    update_res_to_db([str(pred_num+noise), int(te_acc>0.5), 'YJ1'])
    
    time.sleep(0.1)
    
    print(pred_num+noise)
    print(datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S"))
    noise+=1


