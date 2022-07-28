import numpy as np
import cv2.ximgproc as xip
import cv2

def computeDisp(Il, Ir, max_disp):
    # print('max_disp:',max_disp)
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    wdow_size = 7
    radius = wdow_size // 2
    
    pad_Il = cv2.copyMakeBorder(Il, top=radius, bottom=radius, left=radius, right=radius, borderType=cv2.BORDER_CONSTANT,value=0)
    pad_Ir = cv2.copyMakeBorder(Ir, top=radius, bottom=radius, left=radius, right=radius, borderType=cv2.BORDER_CONSTANT,value=0)
    
    slid_wdow_shape = (wdow_size, wdow_size, ch)
    
    v_Il = np.lib.stride_tricks.sliding_window_view(pad_Il, slid_wdow_shape)
    v_Ir = np.lib.stride_tricks.sliding_window_view(pad_Ir, slid_wdow_shape)

    v_Il = v_Il[:,:,0,:,:,:]
    v_Ir = v_Ir[:,:,0,:,:,:]

    census_tfm_Il = np.zeros(shape=v_Il.shape, dtype=int)
    census_tfm_Ir = np.zeros(shape=v_Ir.shape, dtype=int)

    for row in range(v_Il.shape[0]):
        for col in range(v_Il.shape[1]):
            census_tfm_Il[row, col] = ((v_Il[row, col,: ,:, :] - Il[row, col]) >= 0).astype(int)
            census_tfm_Ir[row, col] = ((v_Ir[row, col,: ,:, :] - Ir[row, col]) >= 0).astype(int)


    a_big_number = wdow_size * wdow_size * ch
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    census_cost_l_to_r = np.zeros(shape=(Il.shape[0], Il.shape[1], max_disp),dtype=int)
    census_cost_r_to_l = np.zeros(shape=(Il.shape[0], Il.shape[1], max_disp),dtype=int)
    for col in range(Il.shape[1]):
        for disp in range(max_disp): # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
            #Il to Ir
            if col - disp >= 0:
                #.astype(int) is so important!!!omg
                tmp = (census_tfm_Il[:, col] != census_tfm_Ir[:, col - disp]).astype(int)
                census_cost_l_to_r[:, col, disp] = np.einsum('...ijk->...', tmp)#.astype(np.float32)
            # else:
            #     census_cost_l_to_r[:, col, disp] = a_big_number
            #Ir to Il
            if col + disp < Il.shape[1]:
                #.astype(int) is so important!!!omg
                tmp = (census_tfm_Il[:, col + disp] != census_tfm_Ir[:, col]).astype(int)
                census_cost_r_to_l[:, col, disp] = np.einsum('...ijk->...', tmp)#.astype(np.float32)
            # else:
            #     census_cost_r_to_l[:, col, disp] = a_big_number

    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    for disp in range(1,max_disp):
        census_cost_l_to_r[:,:disp, disp] = census_cost_l_to_r[:,disp+1,disp][..., None]
        census_cost_r_to_l[:,-disp:,disp] = census_cost_r_to_l[:,-(disp+1),disp][..., None]

    # >>> Cost Aggregation
    # Refine the cost according to nearby costs

    # [Tips] Joint bilateral filter (for the cost of each disparty)
    jbf_cost_l_to_r = np.zeros((h, w, max_disp), dtype=np.float32)
    jbf_cost_r_to_l = np.zeros((h, w, max_disp), dtype=np.float32)
    for disp in range(max_disp):
        jbf_cost_l_to_r[:, :, disp] = xip.jointBilateralFilter(Il, census_cost_l_to_r[:, :, disp].astype(np.float32), 15, 10, 10)
        jbf_cost_r_to_l[:, :, disp] = xip.jointBilateralFilter(Ir, census_cost_r_to_l[:, :, disp].astype(np.float32), 15, 10, 10)  

    # >>> Disparity Optimization
    # [Tips] Winner-take-all
    dispmin_l = np.argmin(jbf_cost_l_to_r, axis=2)
    dispmin_r = np.argmin(jbf_cost_r_to_l, axis=2)

    a_big_number = max_disp
    # [Tips] Left-right consistency check
    for row in range(h):
        for col in range(w):
            if(col - dispmin_l[row, col]) >=0:
                if dispmin_l[row, col] != dispmin_r[row, col - dispmin_l[row, col]]:
                    dispmin_l[row, col] = a_big_number
            # else:
            #     print('hey!')


    # [Tips] Hole filling
    for row in range(h):
        for col in range(w):
            if dispmin_l[row, col] == a_big_number:
                current_shift = 0
                left_nothole = a_big_number
                right_nothole = a_big_number
                while((col - current_shift) >= 0):
                    if dispmin_l[row, col - current_shift] != a_big_number:
                        left_nothole = dispmin_l[row, col - current_shift]
                        break
                    else:
                        current_shift += 1
            
                current_shift = 0
                while((col + current_shift) < w):
                    if dispmin_l[row, col + current_shift] != a_big_number:
                        right_nothole = dispmin_l[row, col + current_shift]
                        break
                    else:
                        current_shift += 1

                dispmin_l[row, col] = min(left_nothole, right_nothole)
            # if dispmin_l[row, col] == a_big_number:
            #     print('tooo bad')

    # [Tips] Weighted median filtering
    # print(dispmin_l.dtype)
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), dispmin_l.astype(np.uint8), r = 15)
    return labels.astype(np.uint8)