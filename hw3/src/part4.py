import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping
# import matplotlib.pyplot as plt
random.seed(211)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    
    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching

        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # kp1_coordinate = np.array([i.pt for i in kp1])
        # kp2_coordinate = np.array([i.pt for i in kp1])
        # print(kp1_coordinate[:5])
        # print(kp1_coordinate.shape)
        
        # Draw first 10 matches.
        # img3 = cv2.drawMatches(im1,kp1,im2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.figure(dpi=500)
        # plt.imshow(img3)
        # # plt.show()
        # plt.savefig('./tmpimg.png')

        # TODO: 2. apply RANSAC to choose best H
        # print(matches[0])
        # print(random.sample(matches,4))
        # print(len(matches))

        kp1_matches_coordinate = np.array([kp1[mth.queryIdx].pt for mth in matches])
        kp2_matches_coordinate = np.array([kp2[mth.trainIdx].pt for mth in matches])

        # make perspective projection coordinate
        kp1_ppc = np.concatenate((kp1_matches_coordinate.T, np.ones(shape=(1, len(kp1_matches_coordinate)))), axis=0)
        kp2_ppc = np.concatenate((kp2_matches_coordinate.T, np.ones(shape=(1, len(kp2_matches_coordinate)))), axis=0)
        


    
        # print(kp1_matches_coordinate[:5])

        # print('kp1_ppc.shape:', kp1_ppc.shape)
        # print('kp1_ppc:', kp1_ppc[:,:5])
        max_iter = 10000
        thres = .1
        max_inlier_count = 0
        opti_H = np.zeros(shape=(3,3))
        for i in range(max_iter):
            random_pair = random.sample(matches,4)
            # print('random_pair:', random_pair)
            # print(kp1[random_pair[j].queryIdx].pt for j in range(4))

            rand_img1 = np.array([kp1[j.queryIdx].pt for j in random_pair])
            rand_img2 = np.array([kp2[j.trainIdx].pt for j in random_pair])

            # print('u=',u)
            
            # print('v=',v)
            
            candi_H = solve_homography(rand_img2, rand_img1)
            # print(candi_H.shape)
            kp2_ppc_new = np.dot(candi_H, kp2_ppc)
            kp2_ppc_new = kp2_ppc_new / kp2_ppc_new [-1,:]
            # print(kp2_ppc_new[:,:3])
            diff = np.linalg.norm((kp2_ppc_new - kp1_ppc), axis=0) 
            diff = diff / diff.shape[0]
            inlier_count = np.sum(diff < thres)
            if(inlier_count > max_inlier_count):
                # print(inlier_count)
                # if(inlier_count == 118):
                #     print(kp1_ppc[:,:3])
                #     print(kp2_ppc_new[:,:3])
                max_inlier_count = inlier_count
                opti_H = candi_H
            


        
        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, opti_H)
        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)