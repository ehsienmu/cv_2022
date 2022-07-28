from PIL import Image
import numpy as np
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time
import cv2
from time import time

#This function will sample SIFT descriptors from the training images,
#cluster them with kmeans, and then return the cluster centers.

def build_vocabulary(image_paths, vocab_size):
    ##################################################################################
    # TODO:                                                                          #
    # Load images from the training set. To save computation time, you don't         #
    # necessarily need to sample from all images, although it would be better        #
    # to do so. You can randomly sample the descriptors from each image to save      #
    # memory and speed up the clustering. Or you can simply call vl_dsift with       #
    # a large step size here.                                                        #
    #                                                                                #
    # For each loaded image, get some SIFT features. You don't have to get as        #
    # many SIFT features as you will in get_bags_of_sift.py, because you're only     #
    # trying to get a representative sample here.                                    #
    #                                                                                #
    # Once you have tens of thousands of SIFT features from many training            #
    # images, cluster them with kmeans. The resulting centroids are now your         #
    # visual word vocabulary.                                                        #
    ##################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # This function will sample SIFT descriptors from the training images,           #
    # cluster them with kmeans, and then return the cluster centers.                 #
    #                                                                                #
    # Function : dsift()                                                             #
    # SIFT_features is a N x 128 matrix of SIFT features                             #
    # There are step, bin size, and smoothing parameters you can                     #
    # manipulate for dsift(). We recommend debugging with the 'fast'                 #
    # parameter. This approximate version of SIFT is about 20 times faster to        #
    # compute. Also, be sure not to use the default value of step size. It will      #
    # be very slow and you'll see relatively little performance gain from            #
    # extremely dense sampling. You are welcome to use your own SIFT feature.        #
    #                                                                                #
    # Function : kmeans(X, K)                                                        #
    # X is a M x d matrix of sampled SIFT features, where M is the number of         #
    # features sampled. M should be pretty large!                                    #
    # K is the number of clusters desired (vocab_size)                               #
    # centers is a d x K matrix of cluster centroids.                                #
    #                                                                                #
    # NOTE:                                                                          #
    #   e.g. 1. dsift(img, step=[?,?], fast=True)                                    #
    #        2. kmeans( ? , vocab_size)                                              #  
    #                                                                                #
    # ################################################################################
    '''
    Input : 
        image_paths : a list of training image path
        vocal size : number of clusters desired
    Output :
        Clusters centers of Kmeans
    '''
    """
    Returns
    -------
    frames : `(F, 2)` or `(F, 3)` `float32` `ndarray`
        ``F`` is the number of keypoints (frames) used. This is the center
        of every dense SIFT descriptor that is extracted.
    descriptors : `(F, 128)` `uint8` or `float32` `ndarray`
        ``F`` is the number of keypoints (frames) used. The 128 length vectors
        per keypoint extracted. ``uint8`` by default.
    """

    dsift_step = 3
    print(len(image_paths))
    tmp_img = cv2.imread(image_paths[0], 0)
    _, descriptors = dsift(tmp_img, step=[dsift_step, dsift_step], fast=True)#, float_descriptors=True)
    X = descriptors  
    for img_path in image_paths[1:]:
        tmp_img = cv2.imread(img_path, 0)
        _, descriptors = dsift(tmp_img, step=[dsift_step, dsift_step], fast=True)#, float_descriptors=True)
        
        X = np.concatenate((X, descriptors), axis=0)
       
    # print('shape of X: ', X.shape)
    mtx = X.astype('float32')
    # print('shape of mtx: ', mtx.shape)
    start_time = time()
    vocab = kmeans(X.astype('float32'), vocab_size)
    
    end_time = time()
    # print("kmeans ", (end_time - start_time), " (s).")
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    return vocab



