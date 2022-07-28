from PIL import Image
import numpy as np
import cv2

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    tiny_images = []
    for img_path in image_paths:
        tmp_img = cv2.imread(img_path, 0).astype(np.float32)
        tmp_img = cv2.resize(tmp_img, (16, 16))
        tmp_img_vec = tmp_img.flatten()
        tiny_images.append(tmp_img_vec)
    tiny_images = np.array(tiny_images)
    # print(tiny_images.shape)
    # print(tiny_images.dtype)
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
