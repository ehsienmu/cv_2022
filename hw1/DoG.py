import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold, save_path = ''):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1
        # self.save_dog = save_dog
        self.save_path = save_path

    def get_keypoints(self, image):
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images_octave = []
        tmp_img = image.copy()
        for oct_ind in range(self.num_octaves):
            gaussian_images_octave.append(tmp_img)
            for sig_pow in range(1, self.num_guassian_images_per_octave):
                # print('sigmaX =', self.sigma ** sig_pow)
                gaussian_images_octave.append(cv2.GaussianBlur(tmp_img, ksize = (0, 0), sigmaX = self.sigma ** sig_pow))
            new_width = int(tmp_img.shape[1] // 2)
            new_height = int(tmp_img.shape[0] // 2)
            new_dim = (new_width, new_height)
            if oct_ind != self.num_octaves - 1:
                # print('go next octave')
                tmp_img = cv2.resize(gaussian_images_octave[self.num_guassian_images_per_octave * oct_ind - 1].copy(), new_dim, interpolation = cv2.INTER_NEAREST)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images_octave = []
        for oct_ind in range(self.num_octaves):
            for i in range(self.num_guassian_images_per_octave - 1):
                # dog_images_octave.append(cv2.subtract(gaussian_images_octave[oct_ind * self.num_guassian_images_per_octave + i], gaussian_images_octave[oct_ind * self.num_guassian_images_per_octave + i + 1]))
                dog_images_octave.append(cv2.subtract(gaussian_images_octave[oct_ind * self.num_guassian_images_per_octave + i + 1], gaussian_images_octave[oct_ind * self.num_guassian_images_per_octave + i]))

        # save the images
        if(self.save_path != ''):
            for oct_ind in range(self.num_octaves):
                for i in range(0, self.num_DoG_images_per_octave):
                    img_to_fit = (dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i])
                    tmp_img = (img_to_fit - np.min(img_to_fit)) / (np.max(img_to_fit) - np.min(img_to_fit))
                    cv2.imwrite(self.save_path + 'DoG_' + str(oct_ind + 1) + '-' + str(i + 1) + '.jpg', (tmp_img * 255).astype('uint8'))

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints_list = []
        for oct_ind in range(self.num_octaves):
            for i in range(0 + 1, self.num_DoG_images_per_octave - 1):
                target_img_shape = dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i].shape
                for rows, cols in np.ndindex(target_img_shape[0] - 2, target_img_shape[1] - 2): # from (0, 0) to (w - 3, h - 3)
                    # (candi_r, candi_c) start from (1, 1) to (w - 2, h - 2)
                    candi_r = rows + 1
                    candi_c = cols + 1
                    target_value = dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i][candi_r, candi_c]
                    local_min = True
                    local_max = True
                    if np.abs(target_value) > self.threshold:
                        for shift_r, shift_c in np.ndindex((3,3)):
                            if target_value > dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i - 1][candi_r + shift_r - 1, candi_c + shift_c - 1]:
                                local_min = False
                                break
                            if target_value > dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i][candi_r + shift_r - 1, candi_c + shift_c - 1]:
                                local_min = False
                                break
                            if target_value > dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i + 1][candi_r + shift_r - 1, candi_c + shift_c - 1]:
                                local_min = False
                                break
                        
                        if(local_min == False):
                            for shift_r, shift_c in np.ndindex((3,3)):
                                if target_value < dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i - 1][candi_r + shift_r - 1, candi_c + shift_c - 1]:
                                    local_max = False
                                    break
                                if target_value < dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i][candi_r + shift_r - 1, candi_c + shift_c - 1]:
                                    local_max = False
                                    break
                                if target_value < dog_images_octave[oct_ind * self.num_DoG_images_per_octave + i + 1][candi_r + shift_r - 1, candi_c + shift_c - 1]:
                                    local_max = False
                                    break

                        if (local_min == True) or (local_max == True):
                            # add to keypoint
                            keypoints_list.append((candi_r * (2 ** oct_ind), candi_c * (2 ** oct_ind)))
       
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints_list), axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints