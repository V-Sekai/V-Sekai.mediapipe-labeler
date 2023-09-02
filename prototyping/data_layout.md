# SpeakingFaces Data layout

FLIR T540 thermal camera (464×348 pixels, 24◦ FOV) and a Logitech C920 Pro HD web-camera (768×512 pixels, 78◦ FOV) with a built-in dual stereo microphone were used for data collection purpose.

Each subject participated in two trials where each trial consisted of two sessions. In the first session, subjects were silent and still, with the operator capturing the visual and thermal video streams through the procession of nine collection angles.

The second session consisted of the subject reading a series of commands as presented one-by-one on the video screens, as the visual, thermal and audio data was collected from the same nine camera positions.

Data recording for the first session:

    Launch the MATLAB and start the global ROS node via MATLAB's terminal: rosinit
    Open /record_matlab/record_only_video.m and initialize the following parameters:

    sub_id: a subject ID.
    trial_id: a trial ID.
    numOfPosit: a total number of positions.
    numOfFrames: a total number of frames per position.
    pauseBtwPos: a pause (sec) that is necessary for moving cameras from one position to another.
    path: a path to save the recorded video files.

    Launch /record_matlab/record_only_video.m

Data recording for the second session:

    Launch the MATLAB and start the global ROS node via MATLAB's terminal: rosinit
    Open /record_matlab/record_audio_video.m and initialize the following parameters:

    sub_id: a subject ID.
    trial_id: a trial ID.
    numOfPosit: a total number of positions.
    fpc: a number of frames necessary for reading one character.
    pauseBtwPos: a pause (sec) that is necessary for moving cameras from one position to another.
    path: a path to save the recorded audio and video files.

    Launch /record_matlab/record_audio_video.m

### Step 1: Dataset Preparation

We'll use the [SpeakingFaces](https://github.com/IS2AI/SpeakingFaces) dataset, which contains both audio and corresponding face shapes data.

To calculate the total duration of all audio files in a folder, you can use `soxi` with some shell scripting:

```bash
brew install sox
sum=0; for file in *.wav; do duration=$(soxi -D "$file"); sum=$(echo $sum + $duration | bc); done; echo $sum
```
This script will loop over all `.wav` files in the current directory, get the duration of each file, add this duration to a running total, and then print out the total sum.
