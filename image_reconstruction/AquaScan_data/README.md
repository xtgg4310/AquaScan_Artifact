# Prepare for your artifact evalutaion
This README is written to introduce the format of data, labels, scripts, checkpoints

## Sonar data
Sonar is designed to scan the areas with acoustic echoes. We use Ping360 to collect sonar images. Here we show a sample image of collect sonar data.

![image](https://github.com/xtgg4310/AquaScan_Artifact/blob/main/image_reconstruction/AquaScan_data/figure/sample.png)

Each sonar image will be stored in a txt file which is named with the timestamp that scanning sonar starts to scan. 
Data is orangnized as a 400 * 501 matrix. The first number of one line is the angle (unit: gradian, 1 gradian = 0.9 degree). The 2-501 number represents the amplitude of echoes at corresponding locations.

2292002-2292005 is tested with 10 subjects in the pool yoho. Two sonars are setup at each side of the pool. sonar4 of 2292002 do not receive data so it is empty. There are Label errors in 2292005 sonar11 which degrades the performance of activity recognition(mainly moving) and object detection. We do not include 2292005 sonar11 in evaluatin of object detection since it hard to label one of the moving subject.

## Label
We label the human activity as the three meta-activity: stand, struggle, moving. 

"Moving" is marked when the subjects are swimming in the pool.

"Stand" is marked when the subjects with low-intense motion or completely stillness have slight or minimal location change

"Struggle" is marked when the subjects with high-intense motion have slight or minimal location change. Noted that Subjects marked as Struggle can be normally splashing or struggling.

We labeled the ground truth of five activities as defined in paper Section 4.4. 
* Moving: Subjects marked as "Moving"
* Motionless: Subjects marked as "Stand" and maintain this meta-activity within the threshold of motionless-to-drowning (60 second).
* Splashing: Subjects marked as "Struggle" and maintain this meta-activity within the threshold of splashing-to-Struggling (30 second).
* Struggling: Subjects marked as "Struggle" and maintain this meta-activity over the threshold of splashing-to-Struggling (30 second) but still within the threshold of splashing-to-Struggling (20 second in AquaScan).
* Drowning: (1) Subjects marked as "Stand" and maintain this meta-activity over the threshold of motionless-to-drowning (60 second). (2) Subjects recognized as struggling over the threshold of splashing-to-Struggling (20 second).

### Why we need struggling and two types of drowning
(1) Struggling and drowning can be seen as two stages that one subjects are experiencing. Struggling means subjects have energy when they try to save themselves but drowning means subjects are in extreme danger due to continuous comsuption of energy.

(2) Drowning can be dynamic or static. A long-term motionless can trigger the danger since these static drowning cases are caused by drugs or drunk. We set 60s in the system just for evaluation.

The duration of each activity is defined in Section 5.3. 

## Scripts
The Scripts used for generated experiments results are contained in this folder. 

xxxx_recover.sh: scripts to recover the images
xxxx_label.sh: scripts to run object detection, tracking and moving recognizition. We defined the required paramters in the script.
xxxx_infe.sh: scripts to recognize five activties with state-transfer-machine.

## Checkpoints
This folder contains checkpoint for evaluation.
