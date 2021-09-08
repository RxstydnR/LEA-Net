<< COMMENTOUT

    Note:
        - If you want to save the results of this anomaly detection experiment in the other folder, set the saving folder name at --SAVE_PATH.
        - The architecture of the dataset folder containing CV data have to be as follow:

            --- Dataset --- 0
                        |
                        --- 1
                        :
                        :
                        --- 4
        
COMMENTOUT

dataset="PlantVillage/Potato"
savePath="PlantVillage/Potato"

python anomalydetection.py \
    --dataset     ${dataset} \
    --SAVE_PATH   ${savePath} \
    --ADN         ResNet18 \
    --CAAN        ResNet \
    --attPoint    1 \
    --batchSize   16 \
    --lr          0.0001
