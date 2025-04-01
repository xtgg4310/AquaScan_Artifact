# Prepare for your artifact evaluation
This README is written to introduce the format of data, labels, scripts, checkpoints

## Sonar data
Sonar is designed to scan the areas with acoustic echoes. We use Ping360 to collect sonar images. Here we show a sample image of collect sonar data.

![image](https://github.com/xtgg4310/AquaScan_Artifact/blob/main/image_reconstruction/AquaScan_data/figure/sample.png)

Each sonar image will be stored in a txt file which is named with the timestamp that scanning sonar starts to scan. 
Data is organized as a 400 * 501 matrix. The first number of one line is the angle (unit: gradian, 1 gradian = 0.9 degree). The 2-501 number represents the amplitude of echoes at corresponding locations.

2292002-2292005 is tested with 10 subjects in the pool yoho. Two sonars are set up at each short side of the pool. sonar4 of 2292002 does not receive data so it is empty. There are Label errors (moving, id: 6) in 2292005 sonar11 which degrades the performance of activity recognition(mainly moving) and object detection. We do not include 2292005 sonar11 in the evaluation of object detection since it is hard to label one of the moving subjects.

0807-0821 is tested with 5 subjects in another pool with one sonar setup at the long side of the pool.

## Label
We label a human activity as the three meta-activity: stand, struggle, moving. 

"Moving" is marked when the subjects are swimming in the pool.

"Stand" is marked when the subjects with low-intense motion or complete stillness have a slight or minimal location change

"Struggle" is marked when the subjects with high-intense motion have slight or minimal location change. Noted that Subjects marked as struggling can be normally splashing or struggling.

We labeled the ground truth of five activities as defined in paper Section 4.4. 
* Moving: Subjects marked as "Moving"
* Motionless: Subjects marked as "Stand" and maintaining this meta-activity within the threshold of motionless-to-drowning (60 seconds).
* Splashing: Subjects marked as "Struggle" and maintaining this meta-activity within the threshold of splashing-to-Struggling (30 seconds).
* Struggling: Subjects marked as "Struggle" and maintaining this meta-activity over the threshold of splashing-to-struggling (30 seconds) but still within the threshold of splashing-to-struggling (20 seconds in AquaScan).
* Drowning: (1) Subjects marked as "Stand" and maintaining this meta-activity over the threshold of motionless-to-drowning (60 seconds). (2) Subjects recognized as struggling over the threshold of splashing-to-struggling (20 seconds).

### Why we need struggling and two types of drowning
(1) Struggling and drowning can be seen as two stages that one subject is experiencing. Struggling means subjects have energy when they try to save themselves but drowning means subjects are in extreme danger due to continuous consumption of energy.

(2) Drowning can be dynamic or static. A long-term motionless can trigger danger since these static drowning cases are caused by drugs or drunk. We set 60s in the system just for evaluation.

The duration of each activity is defined in Section 5.3. 

## Scripts
The Scripts used for generated experiment results are contained in this folder. 

xxxx_recover.sh: scripts to recover the images
xxxx_label.sh: scripts to run object detection, tracking, and moving recognition. We defined the required parameters in the script.
xxxx_infe.sh: scripts to recognize five activities with state-transfer-machine.

## Checkpoints
This folder contains a checkpoint for evaluation.

## Metric
Our paper evaluates the five classes' accuracy and F1-score of object detection. 

## Noted
We update prepare.sh in ../image_reconstruction, the old version may lead to miss of checkpoints (optimal.pth). We update the script to fix this error.

