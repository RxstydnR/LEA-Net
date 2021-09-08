import os
import warnings
warnings.simplefilter('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
from utils import get_data, make_color_attention, convert_Lab, data_shuffle
from augmentation import transform
from model.colorization_models import get_colorization_model, EarlyStopping

def colorization(X, color_model_name, batch_size, epochs, SAVE_PATH):
    """ train colorization model
    
        Save:
            history.txt: Colorizatinon network training history
            history.jpg: Colorizatinon network training history
    """

    # get Image Data Generator
    train_datagen = ImageDataGenerator(
        preprocessing_function=transform, 
    )
    train_datagen.fit(X)

    # get model ready
    color_model_name = "Unet"
    model = get_colorization_model(model_name=color_model_name)
    
    print("Colorization training starts")
    # Training
    early_stopper = EarlyStopping(patience=20)
    losses = []
    for epoch in range(epochs):
        batches = 0
        loss = 0
        for X_batch in train_datagen.flow(X, batch_size=batch_size): 
            
            # get Lab images
            X_batch = convert_Lab(X_batch.astype("uint8"))
            X_batch = X_batch.astype("float32")/255.
            
            X_L_batch = X_batch[:,:,:,0:1] # [0,1]
            X_ab_batch = X_batch[:,:,:,1:] # [0,1]

            history = model.fit(X_L_batch, X_ab_batch, verbose=0)
            loss += history.history["loss"][0]
            
            batches += 1
            if batches >= len(X)/batch_size:
                losses.append(loss)

                if (epoch+1)%20==0:
                    print (f"Epoch: {epoch+1}/{epochs}, Loss: {loss}")
                break

        early_stopper(loss)
        if early_stopper.early_stop:
            print (f"Early Stopping")
            break
    
    # Save history
    with open(f"{SAVE_PATH}/history.txt", 'w') as f:
        for l in losses:
            f.write(str(l)+'\n')

    plt.figure(figsize=(8,4))
    plt.plot(losses)
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.title(f'{color_model_name}')
    plt.legend()
    plt.savefig(f"{SAVE_PATH}/history.jpg")
    plt.clf()
    plt.close()
    
    return model


def get_anomaly_map(color_model,X_train,X_test):

    # RGB & L Data
    X_train_Lab = convert_Lab(X_train)
    X_test_Lab  = convert_Lab(X_test)
    
    X_train = X_train.astype('float32')/255.
    X_test  = X_test.astype('float32')/255.
    X_train_Lab = X_train_Lab.astype('float32')/255.
    X_test_Lab  = X_test_Lab.astype('float32')/255.
    
    X_train_L  = X_train_Lab[:,:,:,0:1]
    X_test_L   = X_test_Lab[:,:,:,0:1]
    
    # Colorization
    X_train_color_ab = color_model.predict(X_train_L)
    X_test_color_ab  = color_model.predict(X_test_L)

    # reconstruction
    X_train_color = np.concatenate([X_train_L,X_train_color_ab],axis=-1)
    X_test_color  = np.concatenate([X_test_L, X_test_color_ab],axis=-1)

    # CIEDE2000 attention map
    X_train_att = make_color_attention(X_train_Lab,X_train_color)
    X_test_att  = make_color_attention(X_test_Lab, X_test_color)
    X_train_att = X_train_att[:,:,:,np.newaxis]
    X_test_att  = X_test_att[:,:,:,np.newaxis]
    
    return (X_train,X_test), (X_train_att,X_test_att), (X_train_color,X_test_color)


def main():
    print(f"Path of dataset is {opt.dataset}")

    # get data
    X_P_path = glob.glob(os.path.join(opt.dataset,"Positive","*"))
    X_N_path = glob.glob(os.path.join(opt.dataset,"Negative","*"))
    X_P = get_data(X_P_path)
    X_N = get_data(X_N_path)
    X_P = data_shuffle(X_P)
    X_N = data_shuffle(X_N)
    print(f"Positive:Negative = {len(X_P)}:{len(X_N)}")

    X = np.vstack((X_P,X_N))
    Y = np.array([1]*len(X_P)+[0]*len(X_N)).reshape(-1)
    X,Y = data_shuffle(X,Y)
    
    kfold = StratifiedKFold(n_splits=opt.n_split, shuffle=True, random_state=2020)
    for i, (train, test) in enumerate(kfold.split(X, Y)):
        print(f"{i+1}/{opt.n_split} fold cross validation")

        # make a path to result saving folder
        SAVE_FOLDER = os.path.join(opt.SAVE_PATH,str(i))
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        
        # get train and test data
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        print(f"Train : {X_train.shape, Y_train.shape}")
        print(f"Test  : {X_test.shape, Y_test.shape}")
        
        # get data for colorization model training
        X_train_N = X_train[Y_train==0].astype('float32')/255.

        # train colorozation model
        color_model = colorization(
                        X=X_train_N,
                        color_model_name=opt.model,
                        batch_size=opt.batchSize,
                        epochs=opt.nEpochs,
                        SAVE_PATH=SAVE_FOLDER)
        
        # get attention maps for training and test data
        print("Generate Color Anomaly Maps")
        (X_train,X_test), (X_train_att,X_test_att), (X_train_color, X_test_color) = get_anomaly_map(color_model,X_train,X_test)

        # Save Dataset
        print("Save dataset in .npy format.")
        # RGB
        np.savez(f'{SAVE_FOLDER}/RGB', X_train, X_test)
        # Att
        np.savez(f'{SAVE_FOLDER}/Attention', X_train_att, X_test_att)
        # Colored
        np.savez(f'{SAVE_FOLDER}/Colored', X_train_color, X_test_color)
        # Label
        np.savez(f'{SAVE_FOLDER}/Label', Y_train, Y_test)
        
        # clear cache 
        del color_model
        gc.collect()
        K.clear_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='---')
    parser.add_argument('--dataset',   required=True, help='path to data folder')
    parser.add_argument('--SAVE_PATH', required=True, help='path to save folder')
    parser.add_argument('--n_split',  type=int, default=5,   help='k cross validation".')
    parser.add_argument('--img_size', type=int, default=256, help='image size')

    # colorization training
    parser.add_argument('--model',     type=str, default="Unet", help='colorization model [AE,SegNet,Unet]')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--nEpochs',   type=int, default=500, help='number of epochs to train for')
    # parser.add_argument('--lr',        type=float, default=0.001, help='Learning Rate. Default=0.002')

    opt = parser.parse_args()
    main()

    