# Mobicom2025 Artifact Evaluation: AquaScan: A Sonar-based Underwater Sensing System for Human Activity Monitoring

## This repo contains the code, deployment instructions, Experiment instructions, and detail usage of each function.

## Installation and Preparation for Artifact Evaluation

### Hardware dependencies

Computing platform: A server with CPU and GPU. Our code is tested on a computer with a 7950x3D CPU, 4090 24GB GPU, and 64GB RAM.

Sensor Node:
* Control Unit: Raspberry Pi 4B model 8GB RAM
* Ping360 Scanning Sonar: https://bluerobotics.com/store/sonars/imaging-sonars/ping360-sonar-r1-rp/


### Software dependencies
For common python packages such as numpy, matplotlib, you can download through pip or conda install these packages

Some important package versions:
* numpy version: 1.24.3
* scikit-image: 0.20.0
* scikit-learn: 1.3.0
* matplotlib: 3.7.1
* PyTorch: 2.1.0
* CUDA Version: 12.1
* Opencv-python: 4.8.1.78

If you want to control scanning sonar (Ping 360) to collect data by yourselves, you should install the brping packages by running the command below:
```bash
pip install --user bluerobotics-ping --upgrade
```
### Deployment Instruction
To deploy Ping360 Sonar in the pool, we use a stand shown in the figure below to fix the position of the sonars.

![image](https://github.com/xtgg4310/AquaScan_Artifact/blob/main/figure/setup-2-2.jpg)

### Dataset Setup
Download the dataset from the onedrive link. 

Unzip the file and put raw_data, raw_labels, scripts, checkpoints under the path: ./AquaScan_Artifact/image_reconstruction/AquaScan_data

Run the script ./AquaScan_Artifact/image_reconstruction/prepare.sh
```bash
cd image_reconstruction
bash prepare.sh
```
## Experiments Instruction
### Before Experiments
Please read the README.md in AquaScan_data to learn about the data and the label. Also, it includes the description of scripts.

### Image Reconstruction
Run the script 0229_recover.sh, 0807_recover.sh, 0814_recover.sh, 0821_recover.sh in the folder image_reconstruction

These four scripts will process unreconstructed datasets in the raw_data.

```bash
cd image_reconstruction
bash 0229_recover.sh;bash 0807_recover.sh;bash 0814_recover.sh;bash 0821_recover.sh
```

### Object Detection & Tracking & Moving Detection

Run the script label_0229.sh,label_0807.sh, label_0814.sh,label_0821.sh

For each script label_xxxx.sh, it contains the command to run the pre_sonar_bias.py pre_sonar.py/pre_sonar_opt_yoho.py label2dis.py track.py and moving_detect.py with pre-defined parameters.

The script will generate visualization in the file with the name xxxx_localize_3, intermediate results of tracking track_xxxx and result_xxxx. 

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

### Plot confusion matrix and show the numerical results
```bash
python cal_results.py
```

### Expected Outputs:
Object detection results:

* F1_score: 0.861731843575419
* miss_rate: 0.060167555217060166
* IoU_aver: 0.5205157489569353

Accuracy: 0.916972814107274

### Remove the produced results.

```bash
bash remove_history_0229.sh;bash remove_history_0807.sh;bash remove_history_0814.sh;bash remove_history_0821.sh
bash remove_results.sh
cd image_reconstruction
bash remove_recover.sh
```

## Detail of each function

### Sonar Control
Please see the README.md in Sonar_control folder.

### image reconstruction
Please see the README.md in image_reconstruction folder.

### object detection
We have three codes:
* pre_sonar_bias.py: remove the bias caused by mechanical rotation of sonar (seldom happens)
* pre_sonar.py: denoise sonar images and detect objects on the sonar image.
* pre_sonar_opt_yoho: optimize the dynamic object detection with binary search

#### Usage Guide
#### Parameters of pre_sonar_opt_yoho.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--pre` | int | Yes | preprocess type |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |
| `--label_type` | int | Yes | label type |
| `--parad` | int (multiple values) | Yes | para for noise remove at different distance |
| `--parap` | int (multiple values) | Yes | para for resizing sonar images |
| `--paral` | int (multiple values) | Yes | para for resizing labels |
| `--obj_detect` | str | Yes | save suffix of object detection metric |
| `--obj_type` | str | Yes | folders for object detection metric |
| `--save_dir_all` | str | Yes | the folder that results(non-numerical) saved |

#### Parameters of per_sonar.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--pre` | int | Yes | preprocess type |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |
| `--label_type` | int | Yes | label type |
| `--parad` | int (multiple values) | Yes | para for noise remove at different distance |
| `--parap` | int (multiple values) | Yes | para for resizing sonar images |
| `--paral` | int (multiple values) | Yes | para for resizing labels |
| `--blur_size` | int (multiple values) | Yes | blur size for dynamic processing |
| `--human_size` | int | Yes | human size as the threshold for dynamuic processing (calculate as the bbox size with specific distance) |
| `--remove` | int | Yes | remove semic/static noise under different settings |
| `--bg_path` | str | Yes | background data path |
| `--bg_sc` | str | Yes | backgroud folder name |
| `--max_blur` | int | Yes | max blur size for image denosing |
| `--process` | int | Yes | whether the sonar image is processed by pre_sonar_bias.py |
| `--obj_detect` | str | Yes | save suffix of object detection metric |
| `--obj_type` | str | Yes | folders for object detection metric |
| `--save_dir_all` | str | Yes | the folder that results(non-numerical) saved |

#### Parameters of pre_sonar_bias.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |

### tracking
We have two codes for tracking the objects:
* label2dis.py --generate location of subject in the sonar images
* track.py --generate tracjectory of subjects

#### Usage Guide
#### Parameters of label2dis.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--detect` | str | Yes | label_path |
| `--gt` | str | Yes | gt_path |
| `--type` | int | Yes | data_type |
| `--remove` | int | Yes | data_type |
| `--dis` | int | Yes | dis |
| `--parad` | int (multiple values) | Yes | parad |
| `--save_dir_all` | str | Yes | label_path |

#### Parameters of track.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |
| `--track` | str | Yes | tracking |
| `--track_re` | str | Yes | track_results |
| `--cfg` | int (multiple values) | Yes | cfg |
| `--save_dir_all` | str | Yes | save_dir |

### Moving detection
Moving_detect.py is used for movement detection.

#### Usage Guide
#### Parameters of Moving_detect.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--save` | str | Yes | save_path |
| `--save_dir_all` | str | Yes | save_dir |
| `--pre_cfg` | float (multiple values) | Yes | pre_cfg |
| `--smooth_cfg` | float (multiple values) | Yes | smooth_cfg |

### Generate inference data for motion detection
generate_data_all.py is used for generating inference data

#### Usage Guide
#### Parameters of generate_data_all.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--data` | str | Yes | data_path |
| `--label` | str | Yes | label_path |
| `--save` | str | Yes | save_path |
| `--file` | str | Yes | file |
| `--save_dir_all` | str | Yes | save_dir |

### Recognizing activities through state-transfer-machine
* infe_state.py: motion detection.
* split_results.py: record the motion detection results in separate files.
* state.py: recognize activities. 

#### Usage Guide
#### Parameters of infe_state.py
| Argument Name | Argument Type | Required | Choices | Default Value | Help Information |
| --- | --- | --- | --- | --- | --- |
| `--file_type` | str | No | `yaml`, `json` | `yaml` | None |
| `--option_path` | str | Yes | None | None | path of the option json file |

#### Parameters of split_results.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--dir` | str | Yes | motion |
| `--save_dir` | str | Yes | split_results |
| `--save_dir_all` | str | Yes | moving |

#### Parameters of state.py
| Argument Name | Argument Type | Required | Help Information |
| --- | --- | --- | --- |
| `--motion` | str | Yes | motion |
| `--moving` | str | Yes | moving |
| `--har` | str | Yes | har |
| `--detect` | str | Yes | detect |
| `--label` | str | Yes | detect |
| `--har_cfg` | str | Yes | har_cfg |
| `--smooth_cfg` | str | Yes | smooth_cfg |
| `--start_cfg` | str | Yes | start_cfg |
| `--gt_cfg` | str | Yes | gt_cfg |
| `--gt_mode` | int | Yes | gt_cfg |
| `--gt_trans` | int | Yes | gt_trans |
| `--gt_sc` | str (multiple values) | Yes | gt_trans |
| `--dis` | int | Yes | distance |
| `--label_type` | int | Yes | label_type |
| `--sample` | int | Yes | sample |
| `--save` | str | Yes | save |
| `--save_dir_all` | str | Yes | save_dir_all |

### Show the numerical results and plot the confusion matrix
run the code:
```bash
python cal_results.py
```



