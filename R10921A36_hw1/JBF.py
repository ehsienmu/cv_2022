# from signal import SIG_DFL
import numpy as np
import cv2
# from sklearn.manifold import locally_linear_embedding

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        output = np.zeros_like(img)

        spatial_vec = np.arange(self.wndw_size) - (self.wndw_size-1)//2
        spatial_ratio = (-1.0) / (2 * (self.sigma_s ** 2))
        square_i = (np.dot(spatial_vec.reshape(-1,1), np.ones(shape=(1, spatial_vec.shape[0]))).T ** 2)
        square_j = (np.dot(spatial_vec.reshape(-1,1), np.ones(shape=(1, spatial_vec.shape[0]))) ** 2)
        spatial_kernel = np.exp((square_i + square_j) * spatial_ratio)

        window_shape_gray = (self.wndw_size, self.wndw_size)
        window_shape_color = (self.wndw_size, self.wndw_size, 3)
        if padded_guidance.shape[-1] != 3:
            v_gui = np.lib.stride_tricks.sliding_window_view(padded_guidance, window_shape_gray)
        else:
            v_gui = np.lib.stride_tricks.sliding_window_view(padded_guidance, window_shape_color)
        v_img = np.lib.stride_tricks.sliding_window_view(padded_img, window_shape_color)

        range_ratio = (-1.0)/(2*(self.sigma_r ** 2))
        #  pixel difference: 0 ~ 255
        range_table = np.exp((np.arange(256) / 255) ** 2 * range_ratio)

        # ########################################## v1 & 2   2.4s
        # # center = (self.wndw_size - 1)//2
        # for r, c in np.ndindex(v_gui.shape[0], v_gui.shape[1]):
        #     # tpmtq_abs =  abs(v_gui[r, c, center, center] - v_gui[r, c]) #如果gui是3維前項就該變v_gui[r, c, 0, center, center]才能取到中心點的值
        #     tpmtq_abs =  abs(guidance[r, c] - v_gui[r, c]) # slow?
        #     # print('tpmtq_abs.shape:', tpmtq_abs.shape) # (19, 19) or (1, 19, 19, 3)

        #     if tpmtq_abs.shape[-1] != 3:
        #        range_kernel = range_table[tpmtq_abs] # lookup table
        #     else:
        #        range_kernel = range_table[tpmtq_abs[0,:,:,0]] * range_table[tpmtq_abs[0,:,:,1]] * range_table[tpmtq_abs[0,:,:,2]]
        #     # print('range_kernel.shape:', range_kernel.shape) # (19, 19)
        #     output[r, c] = (np.einsum('ij,ijk->k', np.multiply(spatial_kernel, range_kernel), v_img[r, c, 0])) / np.sum(np.multiply(spatial_kernel, range_kernel))
        # ##########################################

        # ########################################### # v4 2s
        # tmp_gui = np.repeat(guidance[:, :, np.newaxis], self.wndw_size, axis=2)
        # tmp_gui = np.repeat(tmp_gui[:, :, np.newaxis], self.wndw_size, axis=2)
        # if guidance.shape[-1] != 3:
        #     tpmtq_abs = abs(v_gui - tmp_gui) # shape:(316, 316, 19, 19, 3)
        # else:
        #     tpmtq_abs = abs(v_gui[:,:,0,:,:] - tmp_gui) # shape:(316, 316, 19, 19)
            
        # for r, c in np.ndindex(v_gui.shape[0], v_gui.shape[1]):
        #     if guidance.shape[-1] != 3:
        #         range_kernel = range_table[tpmtq_abs[r, c]] # lookup table
        #     else:
        #         range_kernel = range_table[tpmtq_abs[r,c,:,:,0]] * range_table[tpmtq_abs[r,c,:,:,1]] * range_table[tpmtq_abs[r,c,:,:,2]]
        #     output[r, c] = (np.einsum('ij,ijk->k', np.multiply(spatial_kernel, range_kernel), v_img[r, c, 0])) / np.sum(np.multiply(spatial_kernel, range_kernel))
        # ###########################################

        ########################################### v5 1s
        # 把pixel的值塞成window * window 不然不能broadcast
        # 黑白 (316, 316) 變 (316, 316, 19, 19) 
        # 彩色 (316, 316, 3) 變 (316, 316, 19, 19, 3)
        # print('guidance.shape', guidance.shape)
        tmp_gui = np.repeat(guidance[:, :, np.newaxis], self.wndw_size, axis=2)
        # print('tmp_gui.shape', tmp_gui.shape)
        tmp_gui = np.repeat(tmp_gui[:, :, np.newaxis], self.wndw_size, axis=2)
        # print('tmp_gui.shape', tmp_gui.shape)
        
        if guidance.shape[-1] != 3:
            tpmtq_abs = abs(v_gui - tmp_gui)
            range_kernel = range_table[tpmtq_abs] # lookup table
        else:
            tpmtq_abs = abs(v_gui[:,:,0,:,:] - tmp_gui) 
            range_kernel = range_table[tpmtq_abs[:,:,:,:,0]] * range_table[tpmtq_abs[:,:,:,:,1]] * range_table[tpmtq_abs[:,:,:,:,2]]
        # print(range_kernel.shape)
        # print(spatial_kernel.shape)
        sr_arr = np.multiply(spatial_kernel, range_kernel) # shape = (r, c, window, window)
        sum_sr = np.sum(sr_arr, axis=(2,3)) # shape = (r, c)
        output = (np.einsum('...ij,...ijk->...k', sr_arr, v_img[:, :, 0]) / np.repeat(sum_sr[:, :, np.newaxis], img.shape[-1], axis=2)).astype(np.uint8)
        # output = (np.einsum('...ij,...ijk->...k', sr_arr, v_img[:, :, 0]) / np.repeat((np.sum(sr_arr, axis=(2,3)))[:, :, np.newaxis], img.shape[-1], axis=2)).astype(np.uint8)
        ###########################################

        return np.clip(output, 0, 255).astype(np.uint8)