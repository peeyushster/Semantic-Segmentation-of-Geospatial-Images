#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 2 11:51:16 2018

@author: serkankarakulak
"""

import numpy as np
import glob
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import array_ops
import multiprocessing as mp
from multiprocessing.pool import Pool
import random
import math
import os
import time
from tensorflow.python.ops import math_ops

def safe_mkdir(path):
    '''
    Creates a directory if there isn't one already. 
    '''
    try:
        os.mkdir(path)
    except OSError:
        pass

def augmentImage(tup):
    img, augArgs = tup[0], np.array(tup[1])
    augImg = np.rot90(img,k=augArgs[0])
    if (augArgs[1]==1):
        augImg = np.fliplr(augImg)
    if (augArgs[2]==1):
        augImg = np.flipud(augImg)
    if(tup[2]): # for inputs
        augImg= np.true_divide(augImg,127.5) - 1 
        return(augImg)
    else: # for outputs
        tempRtrn =np.zeros_like(augImg)
        for i,v in enumerate([0,17,204]):  # [0,17,204] for [background, buildings, roads]
            tempRtrn[:,:,i] = (augImg[:,:,0]==v).astype('float32')
        return(tempRtrn)

def upsample(x, ratio=2):
    """
    takes a 4D image tensor and increases spatial resolution by replicating values
    so for ratio=2 a 2x2 image becomes a 4x4 image with each value repeated twice
    ratio = # of spatial repeats
    Reference: https://github.com/Newmu/gan_tutorial/blob/master/DCGAN.ipynb
    """
    n_h, n_w = x.get_shape().as_list()[1:3]
    return(tf.image.resize_nearest_neighbor(x, [n_h*ratio, n_w*ratio]))

def mconv(x,hid, kernelSize=3, ind=1,pooling=False,isTraining=True, scope='encoder', reuse = False):
    with tf.variable_scope('mblock{}'.format(ind), reuse=reuse) as scope: 
        conv1 = tf.layers.conv2d(x, hid, kernelSize,name='conv_1',padding='SAME')
        #conv1 = tf.layers.conv2d(x, hid, kernelSize,name='conv_1',padding='SAME')
        conv1 = tf.layers.batch_normalization(conv1, training=isTraining, name='bnorm_1')
        conv1 = tf.nn.leaky_relu(conv1, 0.1, name='lrelu_1')
    return(conv1)

def createConvBlock(x,hid, kernelSize=3, ind=1,pooling=False,isTraining=True, scope='encoder', reuse = False):
    '''
    creates a convolutional layer in the U-Net. 
    
    args:
        x : input layer
        hid : size of the hidden layers
        kernelSize : Receptive filter size of the convolutional layers
        ind : index of the block
        pooling : returns an extra pooling layer after the last layer
        isTraining : True during training, False otherwise
        scope : namescope
        reuse : variable reuse

    return:
        if pooling == True  -> a tuple of last conv layer and a pooling layer
        if pooling == False -> the last conv layer
    '''
    with tf.variable_scope('encConvBlock_{}'.format(ind), reuse=reuse) as scope: 
        if(ind<= 1):
            conv1 = tf.layers.conv2d(x, hid, kernelSize,strides=(2,2),name='conv_1',padding='SAME')
        else:
            conv1 = tf.layers.conv2d(x, hid, kernelSize,name='conv_1',padding='SAME')
        #conv1 = tf.layers.conv2d(x, hid, kernelSize,name='conv_1',padding='SAME')
        conv1 = tf.layers.batch_normalization(conv1, training=isTraining, name='bnorm_1')
        conv1 = tf.nn.leaky_relu(conv1, 0.1, name='lrelu_1')
        conv1 = tf.layers.conv2d(conv1, hid, kernelSize,name='conv_2',padding='SAME')  
        conv1 = tf.layers.batch_normalization(conv1, training=isTraining, name='bnorm_2')
        conv1 = tf.nn.leaky_relu(conv1, 0.1, name='lrelu_2')
        if(pooling):
            pool1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], name='pool', padding='SAME')
    
    if (pooling):
        return(conv1,pool1)
    else:
        return(conv1)
    
    
def uNetLayers(x,numClasses,isTraining=True):
    conv1, pool1 = createConvBlock(x,32,5,ind=1,pooling=True,isTraining=isTraining)        # 128x128x32
    conv2, pool2 = createConvBlock(pool1,64,3,ind=2,pooling=True,isTraining=isTraining)    # 64x64x64
    conv3, pool3 = createConvBlock(pool2,128,3,ind=3,pooling=True,isTraining=isTraining)   # 32x32x128
    conv4, pool4 = createConvBlock(pool3,256,3,ind=4,pooling=True,isTraining=isTraining)   # 16x16x256
    conv5, pool5 = createConvBlock(pool4,512,3,ind=5,pooling=True,isTraining=isTraining)   # 8x8x512
    conv6, pool6 = createConvBlock(pool5,1024,3,ind=6,pooling=True,isTraining=isTraining)  # 4x4x1024
    #conv7, pool7 = createConvBlock(pool6,1024,3,ind=7,pooling=True,isTraining=isTraining)  # 2x2x1024
    upsampleAndMerge = upsampleAndMerge = tf.concat([pool5, upsample(pool6)], -1)          # 4x4x(1024+1024)
    conv8 = createConvBlock(upsampleAndMerge,1024,3,ind=8,isTraining=isTraining)           # 4x4x1024
    conv8 = mconv(conv8,512,3,8,isTraining=isTraining)
    upsampleAndMerge = tf.concat([pool4, upsample(conv8)], -1)                             # 8x8x(1024+512)
    conv9 = createConvBlock(upsampleAndMerge,512,3,ind=9,isTraining=isTraining)            # 8x8x512
    conv9 = mconv(conv9,256,3,9,isTraining=isTraining)
    upsampleAndMerge = tf.concat([pool3, upsample(conv9)], -1)                             # 16x16x(512+256)
    conv10 = createConvBlock(upsampleAndMerge,256,3,ind=10,isTraining=isTraining)          # 16x16x256
    conv10 = mconv(conv10,128,3,10,isTraining=isTraining)
    upsampleAndMerge = tf.concat([pool2, upsample(conv10)], -1)                            # 32x32x(256+128)
    conv11 = createConvBlock(upsampleAndMerge,128,3,ind=11,isTraining=isTraining)          # 32x32x128
    conv11 = mconv(conv11,64,3,11,isTraining=isTraining)
    upsampleAndMerge = tf.concat([pool1, upsample(conv11)], -1)                            # 64x64x(128+64)
    conv12 = createConvBlock(upsampleAndMerge,64,3,ind=12,isTraining=isTraining)           # 64x64x64
    conv12 = mconv(conv12,32,3,12,isTraining=isTraining)
    upsampleAndMerge = tf.concat([conv1, upsample(conv12)], -1)                            # 128x128x(64+32)
    conv13 = createConvBlock(upsampleAndMerge,32,3,ind=13,isTraining=isTraining)           # 128x128x32
    conv13 = mconv(conv13,8,3,13,isTraining=isTraining)
    upsampleAndMerge = upsample(conv13)                                                    # 256x256x32
    conv14 = createConvBlock(upsampleAndMerge,3,3,ind=14,isTraining=isTraining)            # 256x256x3
    preds = tf.nn.softmax(tf.layers.conv2d(conv14, numClasses, 1,name='logits',padding='SAME'))
    return(preds)

    
def focal_loss_softmax(labels,preds,lossW=1., gamma=2,epsilon=1e-8):
    """
    Computer focal loss for multi classification
    
    Args:
      labels : The ground truth output tensor, same dimensions as 'predictions'.
      preds : The predicted outputs.
      lossW : Optional `Tensor` whose rank is either 0, or the same rank as
              `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
              be either `1`, or the same as the corresponding `losses` dimension).
      gamma : A scalar for focal loss gamma hyper-parameter.
      epsilon: A small increment to add to avoid taking a log of zero.
      
    Reference: 
    log_loss() from tensorflow library
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/losses/losses_impl.py
    """
    L = -math_ops.multiply(
        math_ops.multiply(
            labels,
            math_ops.log(tf.clip_by_value(preds, epsilon, 1.0))),
        (1-preds+ epsilon)**gamma
        )  - math_ops.multiply(
        math_ops.multiply(
            (1-labels), math_ops.log(tf.clip_by_value(1-preds, epsilon, 1.0))),
        (preds+ epsilon)**gamma
        )
    return(tf.losses.compute_weighted_loss(L,lossW))



class U_Net(object):
    """
    Implementation of the model
    """
    def __init__(self,
                 lr=0.001,
                 batchSize=32,
                 validBatchSize=10,
                 evalNTimes=4,
                 vers='v0',
                 nProcessesDataPrep=1,
                 testSplit = 0.2,
                 numClasses=3,
                 imgSize=256,
                 skipStep=100,
                 evalAfterStep=0,
                 isTraining = True,
                 noise = 0.1,
                 datasetAddr='/data/sk7685/data1_split',
                 saveCkptAddr='/data/sk7685/MissingMaps',
                 lossWeight_bg = .25,
                 lossWeight_bld = .75,
                 lossWeight_roads = .75
                ):
        
        self.lr = lr
        self.batchSize = batchSize
        self.validBatchSize = validBatchSize
        self.evalNTimes = evalNTimes
        self.vers = vers
        self.nProcessesDataPrep = nProcessesDataPrep
        self.testSplit = testSplit
        self.numClasses = numClasses
        self.imgSize = imgSize
        self.skipStep = skipStep
        self.evalAfterStep = evalAfterStep
        self.isTraining = isTraining
        self.noise = noise
        self.datasetAddr = datasetAddr
        self.saveCkptAddr = saveCkptAddr
        self.lossWeight_bg = lossWeight_bg
        self.lossWeight_bld = lossWeight_bld
        self.lossWeight_roads = lossWeight_roads

        self.lossWeightsArr = np.array([[lossWeight_bg,lossWeight_bld,lossWeight_roads]], dtype='float32')
        self.logFile = 'log_'+ self.vers +'.txt'
        
        self.gstep = tf.Variable(0, 
                                 dtype=tf.int32, 
                                 trainable=False,
                                 name='global_step')
        self.train_x = None
        self.train_y = None
        self.dataPrepPool = mp.Pool(processes=nProcessesDataPrep)

        if(not os.path.isfile('val_' + self.logFile )):
            with open('val_' + self.logFile ,'a') as lgfile:   
                lgfile.write('step\tloss\taccuracy\tIOU_background\tIOU_buildings\tIOU_roads\tPrec_backg\tPrec_build\tPrec_roads\tRecall_backg\tRecall_build\tRecall_roads\tF1_backg\tF1_build\tF1_roads\n')
            with open('valRand_' + self.logFile,'a') as lgfile:   
                lgfile.write('step\tloss\taccuracy\tIOU_background\tIOU_buildings\tIOU_roads\tPrec_backg\tPrec_build\tPrec_roads\tRecall_backg\tRecall_build\tRecall_roads\tF1_backg\tF1_build\tF1_roads\n')


    def trainTestSplit(self, prepTestSet=False):
        '''
        Splits the tiles as test and training sets.
        We can modify the code below to test on selected project IDs.
        '''
        datasetAddr=self.datasetAddr

        if (prepTestSet):
            # testset - projects that are completely reserved for testing
            testFolderNames = []
            for dirPath in glob.glob(datasetAddr+"/Test/*/"):
                testFolderNames.append(dirPath.split('/')[-2])
                
            ##  list containing all the project folders and tile names
            self.test_folderAndTiles = []    
            for f in testFolderNames:
                for image_path in glob.glob(datasetAddr+'/Test/{}/labels/*.png'.format(f)):
                    self.test_folderAndTiles.append( (f, image_path.split('/')[-1][:-4] ) )

            # testset - random tiles that are excluded among the projects
            # that are used in the training
            testRFolderNames = []
            for dirPath in glob.glob(datasetAddr+"/Test_Randomized/*/"):
                testRFolderNames.append(dirPath.split('/')[-2])
                
            ##  list containing all the project folders and tile names
            self.testRand_folderAndTiles = []    
            for f in testRFolderNames:
                for image_path in glob.glob(datasetAddr+'/Test_Randomized/{}/labels/*.png'.format(f)):
                    self.testRand_folderAndTiles.append( (f, image_path.split('/')[-1][:-4] ) )

            self.testDataIndex = 0
            self.testDataSize = len(self.test_folderAndTiles)

            self.testRandomizedDataIndex = 0
            self.testRandomizedDataSize = len(self.testRand_folderAndTiles)
        else:   
            # trainingset - projects that are used in training
            trainFolderNames = []
            for dirPath in glob.glob(datasetAddr+"/Train/*/"):
                trainFolderNames.append(dirPath.split('/')[-2])

            ##  list containing all the project folders and tile names
            self.train_folderAndTiles = []    
            for f in trainFolderNames:
                for image_path in glob.glob(datasetAddr+'/Train/{}/labels/*.png'.format(f)):
                    self.train_folderAndTiles.append( (f, image_path.split('/')[-1][:-4] ) )
                

            # validationset - projects that are completely reserved for validation
            validationFolderNames = []
            for dirPath in glob.glob(datasetAddr+"/Validation/*/"):
                validationFolderNames.append(dirPath.split('/')[-2])
                
            ##  list containing all the project folders and tile names
            self.validation_folderAndTiles = []    
            for f in validationFolderNames:
                for image_path in glob.glob(datasetAddr+'/Validation/{}/labels/*.png'.format(f)):
                    self.validation_folderAndTiles.append( (f, image_path.split('/')[-1][:-4] ) )

            # validationset - random tiles that are excluded among the projects
            # that are used in the training
            validationRFolderNames = []
            for dirPath in glob.glob(datasetAddr+"/Validation_Randomized/*/"):
                validationRFolderNames.append(dirPath.split('/')[-2])
                
            ##  list containing all the project folders and tile names
            self.validationRand_folderAndTiles = []    
            for f in validationRFolderNames:
                for image_path in glob.glob(datasetAddr+'/Validation_Randomized/{}/labels/*.png'.format(f)):
                    self.validationRand_folderAndTiles.append( (f, image_path.split('/')[-1][:-4] ) )


            random.shuffle(self.train_folderAndTiles)
            random.shuffle(self.validation_folderAndTiles)
            random.shuffle(self.validationRand_folderAndTiles)
            self.trainingDataIndex = 0
            self.validationDataIndex = 0
            self.validationRandomizedDataIndex = 0
            self.trainingDataSize = len(self.train_folderAndTiles)
            self.validationDataSize = len(self.validation_folderAndTiles)
            self.validationRandomizedDataSize = len(self.validationRand_folderAndTiles)

        


    def dataGenerator(self,trainingBatch=True):
        """
        Generates new batches of training testing data.
        """
        datasetAddr=self.datasetAddr

        if(trainingBatch):
            if(self.trainingDataIndex+self.batchSize>self.trainingDataSize):
                random.shuffle(self.train_folderAndTiles)
                self.trainingDataIndex = 0                
            
            augParameters = np.hstack([
                np.random.randint(4, size=(self.batchSize,1)), # num of 90 deg rotations
                np.random.randint(2, size=(self.batchSize,1)), # left-right flip
                np.random.randint(2, size=(self.batchSize,1))  # up-down flip
            ])

            self.train_x_new = np.array(
                self.dataPrepPool.map(augmentImage,[(imageio.imread(datasetAddr+'/Train/{0}/tiles/{1}.jpg'.format(t[0],t[1])), augArgs,True)
                 for t, augArgs in 
                 zip(
                    self.train_folderAndTiles[self.trainingDataIndex:self.trainingDataIndex+self.batchSize],
                    augParameters
                    )
                 ]) ,dtype='float32')

            self.train_x_new += np.random.normal(0, self.noise, size=(self.batchSize,self.imgSize,self.imgSize,self.numClasses))

            self.train_y_new = np.array(
                self.dataPrepPool.map(augmentImage,[(imageio.imread(datasetAddr+'/Train/{0}/labels/{1}.png'.format(t[0],t[1])), augArgs,False)
                 for t, augArgs in 
                 zip(
                    self.train_folderAndTiles[self.trainingDataIndex:self.trainingDataIndex+self.batchSize],
                    augParameters
                    )
                 ]),dtype='float32')

            self.trainingDataIndex += self.batchSize
        else:
            # validation set - project completely reserved for validation
            if(self.validationDataIndex+self.validBatchSize>self.validationDataSize):
                random.shuffle(self.validation_folderAndTiles)
                self.validationDataIndex = 0
                
            self.valid_x_new = np.true_divide(np.array(
                [imageio.imread(datasetAddr+'/Validation/{0}/tiles/{1}.jpg'.format(t[0],t[1]))
                 for t in self.validation_folderAndTiles[self.validationDataIndex:self.validationDataIndex+self.validBatchSize]],
            dtype='float32'), 127.5) - 1.
            tmpLabels = np.array(
                [imageio.imread(datasetAddr+'/Validation/{0}/labels/{1}.png'.format(t[0],t[1]))
                 for t in self.validation_folderAndTiles[self.validationDataIndex:self.validationDataIndex+self.validBatchSize]],
            dtype='float32')
            self.valid_y_new =np.zeros_like(tmpLabels)
            for i,v in enumerate([0,17,204]):
                self.valid_y_new[:,:,:,i] = (tmpLabels[:,:,:,0]==v).astype('float32')
            self.validationDataIndex += self.validBatchSize



            # validation set - random tiles that are excluded among the projects
            # that are used in the training 
            if(self.validationRandomizedDataIndex+self.validBatchSize>self.validationRandomizedDataSize):
                random.shuffle(self.validationRand_folderAndTiles)
                self.validationRandomizedDataIndex = 0

            self.validRand_x_new = np.true_divide(np.array(
                [imageio.imread(datasetAddr+'/Validation_Randomized/{0}/tiles/{1}.jpg'.format(t[0],t[1]))
                 for t in self.validationRand_folderAndTiles[self.validationRandomizedDataIndex:self.validationRandomizedDataIndex+self.validBatchSize]],
            dtype='float32'), 127.5) - 1.
            tmpLabels = np.array(
                [imageio.imread(datasetAddr+'/Validation_Randomized/{0}/labels/{1}.png'.format(t[0],t[1]))
                 for t in self.validationRand_folderAndTiles[self.validationRandomizedDataIndex:self.validationRandomizedDataIndex+self.validBatchSize]],
            dtype='float32')
            self.validRand_y_new =np.zeros_like(tmpLabels)
            for i,v in enumerate([0,17,204]):
                self.validRand_y_new[:,:,:,i] = (tmpLabels[:,:,:,0]==v).astype('float32')

            self.validationRandomizedDataIndex += self.validBatchSize

        
        
    def inference(self):
        self.preds = uNetLayers(self.x_ph,self.numClasses,self.isTraining)
            

    def loss(self):
        '''
        Defines loss function of the model
        '''
        # 
        self.lossWeights = tf.placeholder(tf.float32, [1,self.numClasses]) 
        with tf.name_scope('loss'):
            self.y_ph_rs = tf.reshape(self.y_ph,[-1,self.numClasses])
            self.preds_rs = tf.reshape(self.preds,[-1,self.numClasses])
            self.loss = tf.reduce_mean(focal_loss_softmax(self.y_ph_rs,self.preds_rs, self.lossWeights))

    def optimize(self):
        """
        Optimization op
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, 
                                                                   global_step=self.gstep)

    def additionalEvalMetrics(self):
        """
        ! to-do: calculate mean preds among all 90 degree rotations
        """
        with tf.name_scope('prediction_eval'):
            correctPreds = tf.equal(tf.argmax(self.y_ph_rs, 1),tf.argmax(self.preds_rs, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correctPreds,tf.float32))
            
            self.meanIOU,self.IOUconf_mat  = tf.metrics.mean_iou(
                tf.argmax(self.y_ph_rs, -1),
                tf.argmax(self.preds_rs, -1),
                num_classes = 3
            )
            

    def testEval(self, sess, step,evalNTimes):
        self.isTraining = False
        valid_loss_arr = np.zeros((evalNTimes,),dtype='float32')
        valid_mAcc_arr = np.zeros((evalNTimes,),dtype='float32')
#        mIOU_arr = np.zeros((evalNTimes,),dtype='float32')
        valid_confMat_arr = np.zeros((self.numClasses,self.numClasses),dtype='float32')
        
        validRand_loss_arr = np.zeros((evalNTimes,),dtype='float32')
        validRand_mAcc_arr = np.zeros((evalNTimes,),dtype='float32')
#        mIOU_arr = np.zeros((evalNTimes,),dtype='float32')
        validRand_confMat_arr = np.zeros((self.numClasses,self.numClasses),dtype='float32')        

        for i in range(evalNTimes):
            self.valid_x = np.copy(self.valid_x_new)
            self.valid_y = np.copy(self.valid_y_new)
            self.validRand_x = np.copy(self.validRand_x_new)
            self.validRand_y = np.copy(self.validRand_y_new)
            
            pool = Pool(processes=1)
            async_result = pool.apply_async(self.dataGenerator,(False,))
            valid_loss_arr[i], valid_mAcc_arr[i], _ ,valid_tempConf = sess.run(
                    [self.loss,self.accuracy,self.meanIOU,self.IOUconf_mat],
                     feed_dict={self.x_ph: self.valid_x,
                                self.y_ph: self.valid_y,
                                self.lossWeights: self.lossWeightsArr}
                     )
            valid_confMat_arr += valid_tempConf

            validRand_loss_arr[i], validRand_mAcc_arr[i], _ ,validRand_tempConf = sess.run(
                    [self.loss,self.accuracy,self.meanIOU,self.IOUconf_mat],
                     feed_dict={self.x_ph: self.validRand_x,
                                self.y_ph: self.validRand_y,
                                self.lossWeights: self.lossWeightsArr}
                     )
            validRand_confMat_arr += validRand_tempConf            
            pool.close()
            pool.join()


        if (self.logFile):
            IOU_perClass = np.zeros((self.numClasses,),dtype='float32')
            prec_perClass = np.zeros((self.numClasses,),dtype='float32')
            rec_perClass = np.zeros((self.numClasses,),dtype='float32')
            f1_perClass = np.zeros((self.numClasses,),dtype='float32')
            for c in range(self.numClasses):
                IOU_perClass[c] = valid_confMat_arr[c,c] / (valid_confMat_arr[c,:].sum() + valid_confMat_arr[:,c].sum() - valid_confMat_arr[c,c])
                prec_perClass[c] = valid_confMat_arr[c,c] / valid_confMat_arr[:,c].sum() 
                rec_perClass[c] = valid_confMat_arr[c,c] / valid_confMat_arr[c,:].sum() 
                f1_perClass[c] = 2*prec_perClass[c]*rec_perClass[c] / (prec_perClass[c]+rec_perClass[c])
            with open('val_' + self.logFile,'a') as lgfile:
                lgfile.write(
                    '{0}\t{1:.6}\t{2:.6}\t'.format(
                        step,
                        valid_loss_arr.mean(),
                        valid_mAcc_arr.mean()
                    )
                )
                for c in range(self.numClasses):
                    lgfile.write('{0:.6}\t'.format(IOU_perClass[c]))
                for c in range(self.numClasses):
                    lgfile.write('{0:.6}\t'.format(prec_perClass[c]))
                for c in range(self.numClasses):
                    lgfile.write('{0:.6}\t'.format(rec_perClass[c]))
                for c in range(self.numClasses):
                    lgfile.write('{0:.6}\t'.format(f1_perClass[c]))

                lgfile.write('\n')

            for c in range(self.numClasses):
                IOU_perClass[c] = validRand_confMat_arr[c,c] / (validRand_confMat_arr[c,:].sum() + validRand_confMat_arr[:,c].sum() - validRand_confMat_arr[c,c])
                prec_perClass[c] = validRand_confMat_arr[c,c] / validRand_confMat_arr[:,c].sum() 
                rec_perClass[c] = validRand_confMat_arr[c,c] / validRand_confMat_arr[c,:].sum() 
                f1_perClass[c] = 2*prec_perClass[c]*rec_perClass[c] / (prec_perClass[c]+rec_perClass[c])
            with open('valRand_' + self.logFile,'a') as lgfile:
                lgfile.write(
                    '{0}\t{1:.6}\t{2:.6}\t'.format(
                        step,
                        validRand_loss_arr.mean(),
                        validRand_mAcc_arr.mean()
                    )
                )
                for c in range(self.numClasses):
                    lgfile.write('{0:.6}\t'.format(IOU_perClass[c]))
                for c in range(self.numClasses):
                    lgfile.write('{0:.6}\t'.format(prec_perClass[c]))
                for c in range(self.numClasses):
                    lgfile.write('{0:.6}\t'.format(rec_perClass[c]))
                for c in range(self.numClasses):
                    lgfile.write('{0:.6}\t'.format(f1_perClass[c]))

                lgfile.write('\n')            

    def summary(self):
        """
        Summary for TensorBoard
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            #tf.summary.scalar('meanIOU', self.meanIOU)
            self.summary_op = tf.summary.merge_all()

    def build(self):
        """
        Builds the computation graph
        """
        self.x_ph = tf.placeholder(tf.float32, [None, 
                                                self.imgSize,
                                                self.imgSize,
                                                3
                                               ]) 
        self.y_ph = tf.placeholder(tf.float32, [None,
                                                self.imgSize,
                                                self.imgSize,
                                                self.numClasses
                                               ])
        self.trainTestSplit()
        self.dataGenerator()
        self.dataGenerator(trainingBatch=False)
        self.inference()
        self.loss()
        self.optimize()
        self.additionalEvalMetrics()
        self.summary()
    
    def trainOneBatch(self, sess, saver, writer, epoch, step):
#        start_time = time.time()
        self.isTraining = True
        _, l, summaries = sess.run([self.opt, 
                                    self.loss,
                                    self.summary_op],
                                   feed_dict={self.x_ph: self.train_x,
                                              self.y_ph: self.train_y,
                                              self.lossWeights: self.lossWeightsArr})
        writer.add_summary(summaries, global_step=step)
        step += 1
        if(step%1000 == 0):
                saver.save(sess, self.saveCkptAddr+'/checkpoints/missingmaps_'+self.vers+'/cpoint', global_step=step)
#        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def train(self, nBatches):
        """
        Calls the training ops and prepares the training data
        for the next batch in a parallel process.
        """
        safe_mkdir(self.saveCkptAddr)
        safe_mkdir(self.saveCkptAddr+'/checkpoints')
        safe_mkdir(self.saveCkptAddr+'/checkpoints/missingmaps_'+self.vers)
        writer = tf.summary.FileWriter('./graphs/missingmaps_'+self.vers, tf.get_default_graph())

        tVars = tf.trainable_variables()
        defGraph = tf.get_default_graph()

        for v in defGraph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES): 
            if (('bnorm_' in v.name) and
                ('/Adam' not in v.name) and
                ('Adagrad' not in v.name) and
                (v not in tVars )):
                tVars.append(v)
                
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver = tf.train.Saver(var_list= tVars,max_to_keep=2)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.saveCkptAddr+'/checkpoints/missingmaps_'+self.vers+'/cpoint'))

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()

            for epoch in range(nBatches):
                self.train_x = np.copy(self.train_x_new)
                self.train_y = np.copy(self.train_y_new)
                
                pool = Pool(processes=1)
                async_result = pool.apply_async(self.dataGenerator,())
                step = self.trainOneBatch(sess, saver, writer, epoch, step)
                pool.close()
                pool.join()

                if ( ((step + 1) % self.skipStep == 0) and (step>self.evalAfterStep ) ) :
                    self.testEval(sess,step, self.evalNTimes)


        writer.close()
        self.isTraining = False
        #self.dataPrepPool.join()
        #self.dataPrepPool.close()

