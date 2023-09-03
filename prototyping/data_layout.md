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

### Layout

The schema you provided appears to be a list of audio files in a directory structure. Here's how it can be decoded:

- `10/sub_10_ia/trial_1/mic1_audio_cmd_trim/`: This is the directory path where the audio files are stored. It suggests that these files belong to subject 10, trial 1, and were recorded using microphone 1.

- `10_1_2_1_1018_1.wav`: This is an example of one of the audio files. The filename seems to follow a specific pattern which can be broken down as follows:

    - `10`: This could represent the subject number.
    - `1`: This could represent the trial number.
    - `2`: This is unclear without more context, but it could possibly represent a session or task number within the trial.
    - `1`: This again is unclear without more context, but it could possibly represent a sub-task or activity number within the session.
    - `1018`: This could be an identifier for the specific command or action performed.
    - `1`: This could represent a version or take number.
    - `.wav`: This is the file extension, indicating that this is a WAV audio file.

The new list of files you provided is consistent with the previously decoded schema, but it seems to be a list of image files instead of audio files. Here's how it can be decoded:

- `10/sub_10_ia/trial_1/rgb_image_cmd_aligned/10_1_2_`: This is the directory path where the image files are stored. It suggests that these files belong to subject 10, trial 1, and were captured during session or task 2.

- `10_1_2_1_1018_10_3.png`: This is an example of one of the image files. The filename seems to follow a specific pattern which can be broken down as follows:

    - `10`: This represents the subject number.
    - `1`: This represents the trial number.
    - `2`: This could possibly represent a session or task number within the trial.
    - `1`: This could possibly represent a sub-task or activity number within the session.
    - `1018`: This could be an identifier for the specific command or action performed.
    - `10`: This could represent a frame number within the sequence of images captured for this particular action.
    - `3`: This could represent a version or take number.
    - `.png`: This is the file extension, indicating that this is a PNG image file.
    
### Step 1: Dataset Preparation

We'll use the [SpeakingFaces](https://github.com/IS2AI/SpeakingFaces) dataset, which contains both audio and corresponding face shapes data.

To calculate the total duration of all audio files in a folder, you can use `soxi` with some shell scripting:

```bash
brew install sox
sum=0; for file in *.wav; do duration=$(soxi -D "$file"); sum=$(echo $sum + $duration | bc); done; echo $sum
```
This script will loop over all `.wav` files in the current directory, get the duration of each file, add this duration to a running total, and then print out the total sum.
