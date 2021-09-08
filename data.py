import cv2
import numpy as np
from utils import convert_RGB

def load_data(path):

    X_rgb = np.load(f"{path}/RGB.npz")
    X_att = np.load(f"{path}/Attention.npz")
    X_cld = np.load(f"{path}/Colored.npz")
    Y     = np.load(f"{path}/Label.npz")

    X_rgb_train,X_rgb_test = X_rgb['arr_0'],X_rgb['arr_1']
    X_att_train,X_att_test = X_att['arr_0'],X_att['arr_1']
    X_cld_train,X_cld_test = X_cld['arr_0'],X_cld['arr_1']
    Y_train,Y_test = Y['arr_0'],Y['arr_1']

    # Lab -> RGB
    X_cld_train = convert_RGB(X_cld_train)
    X_cld_test = convert_RGB(X_cld_test)

    return (X_rgb_train,X_rgb_test),(X_att_train,X_att_test),(X_cld_train,X_cld_test),(Y_train,Y_test)


def exclude_positive(X_rgb,X_att,Y,rate_P):
    
    len_P = len(Y[Y==1])

    X_rgb_P = X_rgb[Y==1]
    X_rgb_N = X_rgb[Y==0]
    X_att_P = X_att[Y==1]
    X_att_N = X_att[Y==0]

    X_rgb_P_use = X_rgb_P[:int(len_P*rate_P)]
    X_att_P_use = X_att_P[:int(len_P*rate_P)]

    X_rgb_new = np.concatenate([X_rgb_N, X_rgb_P_use])
    X_att_new = np.concatenate([X_att_N, X_att_P_use])
    Y_new = np.concatenate([np.zeros(len(X_rgb_N)),np.ones(len(X_rgb_P_use))])
    
    p = np.random.permutation(len(X_rgb_new))
    X_rgb_new, X_att_new, Y_new = X_rgb_new[p], X_att_new[p], Y_new[p]
    
    return X_rgb_new, X_att_new, Y_new


def resize(X_att,size=8):
    X_resized=[]
    for i in range(len(X_att)):
        x_att = X_att[i]
        org_shape = x_att.shape[:-1]
        x_resized = cv2.resize(x_att,(size,size),cv2.INTER_NEAREST)
        x_re_resized = cv2.resize(x_resized,org_shape,cv2.INTER_NEAREST)
        X_resized.append(x_re_resized)
    X_resized = np.array(X_resized)
    X_resized = np.expand_dims(X_resized,axis=-1)
    
    return X_resized


def normalize(X_att):
    X_norm=[]
    for i in range(len(X_att)):
        x_att = X_att[i]
        att_max = np.max(x_att,axis=(0,1))
        x_norm = x_att / att_max
        X_norm.append(x_norm)
    X_norm = np.array(X_norm)
    
    return X_norm


def binarize(X_att,threshold=0.2):
    X_bin=[]
    for i in range(len(X_att)):
        x_att = X_att[i]
        x_bin = np.where(x_att>threshold,1,0)
        X_bin.append(x_bin)
    X_bin = np.array(X_bin)
    
    return X_bin
