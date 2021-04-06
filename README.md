# Predict Car Speed from Dashcam video (Odometry)

## Goal

_Basically, the goal is to predict the speed of a car from a video._

data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
data/train.txt contains the speed of the car at each frame, one speed on each line.

data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.
The deliverable is test.txt

## Usage:  
#### For Train
```bash
python main.py data/train.mp4 data/train.txt --mode=train --split=0.3
```
If you'd like to continue training using the pretrained network then add the `--resume` flag to that line. <br>
If any modifications are made to the optical flow part of the model then `--wipe` must be used to reprocess the data

#### For Evaluate
```bash
python main.py data/train.mp4 data/train.txt --mode=eval
```
This will print the mean squared error.

#### For Play
```bash
python main.py data/train.mp4 data/train.txt --mode=play
```
If you want a more graphical display you can use the play mode. This will output the Optical Flow video with prediction overlay.

#### For Test/Inference
```bash
python main.py data/test.mp4 data/test.txt --mode=test
```
It will infer the model and save the predicted value to test.txt file.

## Results
I divided flow_data in 70-30 ratio in train and validation set. <br>
I got a MSE of around 2.5 on train data and 0.45 on validation data. MSE on entire train data is **0.55**.

#### Visualization of Results
![output1](results/output.gif) <br>

![output2](results/output2.gif)
