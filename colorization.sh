<< COMMENTOUT

    Note:
        - If you want to save Cross Validation dataset in other folder, set other saving folder name at --SAVE_PATH.
        - Dataset folder specified in --dataset has to contain "Positive" and "Negative" folders where positive and negative images are stored, respectively.
        
        Ex) 
            --- Dataset --- Positive
                        |
                        --- Negative
                        
COMMENTOUT

dataset="PlantVillage/Potato"
savePath="PlantVillage/Potato"

python colorization.py --dataset ${dataset} --SAVE_PATH ${savePath}