# Mobicom2025 Artifact Evaluation: AquaScan: A Sonar-based Underwater Sensing System for Human Activity Monitoring

## This repo contains the code, deployment instructions, Experiment instruction and detail usage of each function.

## Prepare for Artifact evaluation
### Software dependencies
For common python packages such as numpy, matplotlib, you can download through pip or conda install these packages

Some important package version:
numpy version: 1.24.3
scikit-image: 0.20.0
scikit-learn: 1.3.0
matplotlib: 3.7.1
PyTorch: 2.1.0
CUDA Version: 12.1
Opencv-python: 4.8.1.78


### Dataset Setup
Download the dataset from the onedrive link. 

Unzip the file and put raw_data, raw_labels, scripts, checkpoints under the path: ./AquaScan_Artifact/image_reconstruction/AquaScan_data

Run the script ./AquaScan_Artifact/image_reconstruction/prepare.sh
```bash
cd image_reconstruction
bash prepare.sh
```
## Experiments Instruction

### Image Reconstruction
Run the script 0229_recover.sh, 0807_recover.sh, 0814_recover.sh, 0821_recover.sh in the folder image_reconstruction
```bash
bash 0229_recover.sh;bash 0807_recover.sh;bash 0814_recover.sh;bash 0821_recover.sh
```

### Object Detection & Tracking & Moving Detection
Run the script label_0229.sh,label_0807.sh, label_0814.sh,label_0821.sh
```bash
cd ..
bash label_0229.sh;bash label_0807.sh;bash label_0814.sh;bash label_0821.sh
```

### Activity Recognition
Run the script infe_0229.sh,infe_0807.sh, infe_0814.sh,infe_0821.sh
```bash
bash create_folder.sh
bash infe_0229.sh;bash infe_0807.sh;bash infe_0814.sh;bash infe_0821.sh
```

### Plot Confusion Matrix and Show the numercial results
```bash
python cal_results.py
```

### Remove the produced results.

```bash
bash remove_history_0229.sh;bash remove_history_0807.sh;bash remove_history_0814.sh;bash remove_history_0821.sh
bash remove_results.sh
cd image_reconstruction
bash remove_recover.sh
```

### Expected Outputs:
F1-score:

