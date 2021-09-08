import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_ubyte
from skimage.color import deltaE_ciede2000


def get_data(Imgs, img_size=256, BW=False):
    X = []
    for img in Imgs:
        x = cv2.imread(img)
        x = cv2.resize(x, (img_size,img_size))
        if BW: 
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = x[:,:,np.newaxis]
        else:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        X.append(x)
    X = np.array(X)
    
    if X.shape[-1]>3:
        X = X[...,:3]
    
    return X

def convert_RGB(Imgs):
    X = []
    for x in Imgs:
        x = img_as_ubyte(x)
        x = cv2.cvtColor(x, cv2.COLOR_Lab2RGB)
        x = x.astype(np.float32)/255.
        X.append(x)
    return np.array(X)

def convert_Lab(Imgs):
    X = []
    for x in Imgs:
        assert x.dtype == np.uint8,"uint data is expected."

        x = cv2.cvtColor(x, cv2.COLOR_RGB2Lab)
        X.append(x)
    return np.array(X)


def data_shuffle(X,Y=[]):
    ''' Shuffle data
    '''   
    # np.random.seed(0)
    if len(Y)>0:
        assert len(X)==len(Y), "length of X and Y is not same."
        p = np.random.permutation(len(X))
        return X[p],Y[p]
    else:
        p = np.random.permutation(len(X))
        return X[p]


def make_color_attention(X_org, X_cld, params=(1,1,1)):
    '''Calculate ciede2000 between original image and pix2pix colored image, and make the heatmap
    
    Args:
        X_org: Array of RGB Images
        X_cld: Array of RGB Images colored by pix2pix 
    Return:
        X_cie: Array of CIEDE2000 heatmap
    
    '''
    X_cie = []
    for i in range(len(X_org)):
        # make image
        x_org = X_org[i]
        x_cld = X_cld[i]
        x_org_lab = cv2.cvtColor(x_org, cv2.COLOR_RGB2Lab)
        x_cld_lab = cv2.cvtColor(x_cld, cv2.COLOR_RGB2Lab)
        
        # CIEDE2000 heatmap
        att = deltaE_ciede2000(x_org_lab, x_cld_lab, kL=params[0], kC=params[1], kH=params[2])
        X_cie.append(att)
    return np.array(X_cie)


def plot_history(history, path, model_name, fold):
    
    # Accuracy
    plt.figure()
    history.filter(like='acc', axis=1).plot()
    plt.title('Training Accracy')
    plt.savefig(os.path.join(path,f"history_acc_{model_name}_{fold}.jpg"))
    plt.show()
    
    # Loss
    plt.figure()
    try:
        history.filter(like='_loss', axis=1).plot()
    except:
        history.filter(like='loss', axis=1).plot()
    plt.title('Training Loss')
    plt.savefig(os.path.join(path,f"history_loss_{model_name}_{fold}.jpg"))
    plt.show()
    
    plt.clf(),plt.close()
    return
