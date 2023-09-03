# Repository Structure and Data Description

This repository contains annotated data (metadata), raw data, and clean data. The structure of the repository is illustrated in Figure 7a.

## Notation

The names of directories and files follow a specific notation:

- `subID = {1 . . . 142}`: Subject number.
- `trialID = {1, 2}`: Trial number.
- `sessionID`: 1 if the session does not involve utterances, 2 otherwise.
- `posID = {1 . . . 9}`: Camera position.
- `commandID = {1 . . . 1,297}`: Command number.
- `frameID = {1 . . . 900}`: Number of an image in a sequence.
- `streamID`: 1 for thermal images, 2 for visual images, and 3 for the aligned version of the visual images.
- `micID`: 1 for the left microphone and 2 for the right microphone on the web camera.

## Annotated Data

The annotated data are stored in the `metadata` directory, which consists of the `subjects.csv` file and the `commands` subdirectory.

- `subjects.csv`: Contains information on the ID, split (train/valid/test), gender, ethnicity, age, and accessories (hat, glasses, etc.) in both trials for each subject.
- `commands` subdirectory: Consists of `sub_subID_trial_trialID.csv`, composed of records on each command uttered by the subject `subID` in the trial `trialID`. There are 284 files in total, two files for each of the 142 subjects. A record includes the command name, the command identifier, the identifier of a camera position at which the utterance was captured, the transcription of the uttered command, and information on the artifacts detected in the recording.

## Raw and Clean Data

The raw and clean data are organized in a directory structure that follows the above notation. For example, the path `10/sub_10_ia/trial_1/mic1_audio_cmd_trim/` leads to audio files for subject 10, trial 1, recorded using microphone 1. Similarly, `10/sub_10_ia/trial_1/rgb_image_cmd_aligned/10_1_2_` leads to aligned RGB images for subject 10, trial 1, during session 2.

Each file name also follows the notation. For instance, `10_1_2_1_1018_1.wav` is an audio file for subject 10, trial 1, session 2, camera position 1, command 1018, version 1. Similarly, `10_1_2_1_1018_10_3.png` is an image file for subject 10, trial 1, session 2, camera position 1, command 1018, frame 10, version 3.

## Frame Rate

The frame rate for the video data is 28 frames per second.