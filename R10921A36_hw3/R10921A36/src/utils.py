import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    # print(u)
    # print(v) 
    N = u.shape[0] # N=4
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    # u
    # [[  0   0]                                  
    # [512   0]                                  
    # [512 748]                                  
    # [  0 748]]   
    # v                              
    # [[749 521]                                  
    # [883 525]                                  
    # [883 750]                                  
    # [750 750]]  
    # print(u*v)
    # print((u*v)[:,1])
    # print(v[:,[1,0]])
    # (u*v) = (uxvx, uyvy)
    # (u*v[:,[1,0]) = (uxvy, uyvx)
    uv = u*v
    uxvx = uv[:,0]
    uyvy = uv[:,1]

    uvexg = u*v[:,[1,0]]
    uxvy = uvexg[:,0]
    uyvx = uvexg[:,1] # shape is (4,)

    # print(uyvx)
    xeq = np.concatenate((u, np.ones(shape=(N, 1)), np.zeros(shape=(N, 3)), -1 * uxvx[..., None], -1 * uyvx[..., None], -1 * (v[:,0])[..., None]), axis=1)
    yeq = np.concatenate((np.zeros(shape=(N, 3)), u, np.ones(shape=(N, 1)), -1 * uxvy[..., None], -1 * uyvy[..., None], -1 * (v[:,1])[..., None]), axis=1)
    A = np.concatenate((xeq, yeq), axis=0)
    # print(A.shape) # (8,9)

    # TODO: 2.solve H with A
    _, _, vt = np.linalg.svd(A)
    H = vt[-1, :]
    # print(np.linalg.norm(vt[-1, :]))
    # H = vt[-1, :] / np.linalg.norm(vt[-1, :])
    # print(H)        
    # print( np.linalg.norm(H))
    return H.reshape(3, 3)


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    
    xv, yv = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1))

    xv_flatten = xv.flatten()
    yv_flatten = yv.flatten()
    one_flatten = np.ones(shape=(xv_flatten.shape))
    
    mat = np.array([xv_flatten, yv_flatten, one_flatten])

    if direction == 'b':
       
        H_inv = np.linalg.inv(H)
        new_mat = np.dot(H_inv, mat)
        new_mat = new_mat / new_mat[-1, :]
        new_mat = np.round(new_mat)
        mask = (new_mat[0,:] >= 0) & (new_mat[0,:] < w_src) & (new_mat[1,:] >= 0) & (new_mat[1,:] < h_src)

        valid_src_x = (new_mat[0,:].astype(int))[mask]
        valid_src_y = (new_mat[1,:].astype(int))[mask]
        valid_dst_x = (mat[0,:].astype(int))[mask]
        valid_dst_y = (mat[1,:].astype(int))[mask]
        
    elif direction == 'f':
        new_mat = np.dot(H, mat)
        new_mat = new_mat / new_mat[-1, :]
        new_mat = np.round(new_mat)
        mask = (new_mat[0,:] >= 0) & (new_mat[0,:] < w_dst) & (new_mat[1,:] >= 0) & (new_mat[1,:] < h_dst)
        
        valid_src_x = (mat[0,:].astype(int))[mask]
        valid_src_y = (mat[1,:].astype(int))[mask]
        valid_dst_x = (new_mat[0,:].astype(int))[mask]
        valid_dst_y = (new_mat[1,:].astype(int))[mask]

    dst[valid_dst_y, valid_dst_x] = src[valid_src_y, valid_src_x]
    return dst
