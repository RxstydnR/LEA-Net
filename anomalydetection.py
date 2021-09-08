import os
import warnings
warnings.simplefilter('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,f1_score

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from utils import plot_history
from data import load_data, exclude_positive, normalize
from model.LEANet import VGG16,ResNet18,CNN


def preparation(config, X_rgb_train, X_rgb_test, X_att_train, X_att_test, Y_train, Y_test):

    img_shape=(256,256,3)
    att_shape=(256,256,1)
    att_points=config["AttPoint"]

    X_att_train,X_att_test = normalize(X_att_train),normalize(X_att_test)  # Max Normalize
    
    X_fit_train = [X_rgb_train, X_att_train]
    X_fit_test = [X_rgb_test, X_att_test]

    if config["CAAN"]=="Direct":
        config["outputs"] = "oneway"
        Y_fit_train = to_categorical(Y_train,2)
        Y_fit_test = to_categorical(Y_test,2)
    else:
        config["outputs"] = "separate"
        Y_fit_train = [to_categorical(Y_train,2),to_categorical(Y_train,2)]
        Y_fit_test = [to_categorical(Y_test,2),to_categorical(Y_test,2)]
    
    # Model
    if config["ADN"]=="VGG16":
        
        model = VGG16(
            img_shape=img_shape,
            att_shape=att_shape,
            att_points=att_points,
            input_method  = "none",
            distil_method = "avg",
            sigmoid_apply = True, 
            fusion_method = "attention",
            flat_method = "flat",
            att_base_model=config["CAAN"],
            output_method=config["outputs"],
        ).build()

    elif config["ADN"]=="ResNet18":
        
        model = ResNet18(
            img_shape=img_shape,
            att_shape=att_shape,
            att_points=att_points,
            input_method  = "none",
            distil_method = "avg",
            sigmoid_apply = True, 
            fusion_method = "attention",
            flat_method = "gap",
            att_base_model=config["CAAN"],
            output_method=config["outputs"],
        ).build()

    elif config["ADN"]=="CNN":

        model = CNN(
            img_shape=img_shape,
            att_shape=att_shape,
            att_points=att_points,
            input_method  = "none",
            distil_method = "avg",
            sigmoid_apply = True, 
            fusion_method = "attention",
            flat_method = "flat",
            output_method = config["outputs"]
        ).build()
    
    return (X_fit_train,Y_fit_train),(X_fit_test,Y_fit_test), model


def main():

    print(f"Path of dataset is {opt.dataset}")

    n_CV = 5
    for i in range(n_CV): # 5-fold CV
        print(f"{i+1}/{n_CV} fold cross validation")

        # load data
        dataset_path = os.path.join(opt.dataset,str(i))
        (X_rgb_train,X_rgb_test),(X_att_train,X_att_test),(X_cld_train,X_cld_test),(Y_train,Y_test) = load_data(dataset_path)

        # Exclude positive data
        if opt.p_rate < 1.0:
            before_len_p = len(Y_train[Y_train==1])
            print(f"Exclude positive data.")
            X_rgb_train,X_att_train,Y_train = exclude_positive(X_rgb_train,X_att_train,Y_train,rate_P=opt.p_rate)
            after_len_p = len(Y_train[Y_train==1])
            print(f"From {before_len_p} to {after_len_p}.")

        # remove unnecessary arrays 
        del X_cld_train
        del X_cld_test

        # make a path to result saving folder
        SAVE_FOLDER = os.path.join(opt.SAVE_PATH,str(i))
        os.makedirs(SAVE_FOLDER, exist_ok=True)

        # set LEA-Net config
        config={
            "AttPoint":[opt.attPoint-1],
            "ADN":opt.ADN,
            "CAAN":opt.CAAN,
        }
        
        # get model
        (X_fit_train,Y_fit_train),(X_fit_test,Y_fit_test), model = preparation(config,X_rgb_train,X_rgb_test,X_att_train,X_att_test,Y_train,Y_test)
        model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=opt.lr), metrics=["accuracy"])

        # Training
        print("Training ...")
        history = model.fit(
                X_fit_train, Y_fit_train, 
                batch_size=opt.batchSize,
                epochs=opt.nEpochs,
                verbose=2)
        
        # Save model
        model_name = f"ADN-{opt.ADN}_CAAN-{opt.CAAN}"
        model.save(f'{SAVE_FOLDER}/{model_name}_{i}.h5', include_optimizer=False)
        
        # Save history
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(f'{SAVE_FOLDER}/history_{model_name}_{i}.csv')
        plot_history(hist_df, path=SAVE_FOLDER, model_name=model_name, fold=i)

        # Evaluation
        pred = model.predict(X_fit_test)

        print("Evaluation ...")
        if opt.CAAN == "Direct":

            pred_ADN = np.argmax(pred, axis=1)
            pred_CAAN = np.zeros_like(pred_ADN)
            
            print(f"Confusion Matrix \n{confusion_matrix(Y_test, pred_ADN)}")
            print(f"Multiple Scores \n{classification_report(Y_test, pred_ADN)}")
            print(f"F score = {f1_score(Y_test,pred_ADN)}")

        else:
            pred_ADN = np.argmax(pred[0], axis=1)
            pred_CAAN = np.argmax(pred[1], axis=1)
            
            print("========== ADN (Anomaly Detection Network) ==========")
            print(f"Confusion Matrix \n{confusion_matrix(Y_test, pred_ADN)}")
            print(f"Multiple Scores \n{classification_report(Y_test, pred_ADN)}")
            print(f"F score = {f1_score(Y_test,pred_ADN)}")
            
            print("========== CAAN (Color Anomaly Attention Network) ==========")
            print(f"Confusion Matrix \n{confusion_matrix(Y_test, pred_CAAN)}")
            print(f"Multiple Scores \n{classification_report(Y_test, pred_CAAN)}")
            print(f"F score = {f1_score(Y_test,pred_CAAN)}")

        # Save predictions and true labels
        pred_label_df = pd.DataFrame(columns=["pred_ADN","pred_CAAN","true"])
        pred_label_df["pred_main"] = pred_ADN
        pred_label_df["pred_att"] = pred_CAAN
        pred_label_df["true"] = Y_test
        pred_label_df.to_csv(f'{SAVE_FOLDER}/prediction_{model_name}_{i}.csv')
        
        del model
        K.clear_session()
        gc.collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='---')
    parser.add_argument('--dataset', required=True, help='path to data folder')
    parser.add_argument('--SAVE_PATH',required=True, help='path to save folder')
    parser.add_argument('--p_rate', type=float, default=1.0, help='Ratio of Positive data in the training dataset.')

    # anomaly detection
    parser.add_argument('--ADN', type=str, default="VGG16", choices=["VGG16","ResNet18","CNN"], help='Anomaly Detection Network (ADN in the paper).')
    parser.add_argument('--CAAN', type=str, default="MobileNet", choices=["MobileNet","ResNet","Direct"],help='External Network: Color Anomaly Attention Network (CAAN in the paper).')
    parser.add_argument('--attPoint', type=int, default=1, choices=[1,2,3,4,5])
    parser.add_argument('--batchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--nEpochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate for LEA-Net.')
    opt = parser.parse_args()

    print(f"ADN is {opt.ADN}")
    print(f"CAAN is {opt.CAAN}")

    main()

    