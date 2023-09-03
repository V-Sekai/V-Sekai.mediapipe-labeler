#!/usr/bin/env python3
import os
import glob
import subprocess
import argparse
import re
import tempfile
import numpy as np

class FileStruct:
    def __init__(self):
        self.wave_file = None
        self.png_files = []

    def add_wave(self, wave_file):
        self.wave_file = wave_file

    def add_png(self, png_file):
        self.png_files.append({"png_file": png_file, "wave_file": self.wave_file})

    def print(self):
        print(f"Wave file: {self.wave_file}")
        print("PNG files:")
        for png_dict in self.png_files:
            print(f"PNG file: {png_dict['png_file']}, Wave file: {png_dict['wave_file']}")


def get_script_dir():
    return os.path.dirname(os.path.realpath(__file__))

def process_directory(dir):
    if not os.path.isdir(dir):
        return
    print(f"Processing {dir}")
    print("Directory contents before processing:")
    print(os.listdir(dir))
    
    # Create an instance of FileStruct
    file_struct = FileStruct()

    for trial_dir in glob.glob(os.path.join(dir, 'trial_*')):
        # Process each trial directory and get the filled FileStruct
        trial_file_struct = process_trial(trial_dir)
        
        # If there is a wave file in the trial directory, add it to the main FileStruct
        if trial_file_struct.wave_file:
            file_struct.add_wave(trial_file_struct.wave_file)
        
        # Sort png_files based on sequence number extracted from filename
        trial_file_struct.png_files.sort(key=lambda f: int(re.search(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).png$', f['png_file']).group(5)) if re.search(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).png$', f['png_file']) else 0)
                
        # Add all png files from the trial directory to the main FileStruct
        for i, png_file in enumerate(trial_file_struct.png_files):
            file_struct.add_png(png_file)

    return file_struct

def sort_key_func(item):
    parts = item.split("_")
    
    # Ensure there are enough parts in the filename
    if len(parts) < 7:
        return [0]*7

    keys = []
    for part in parts[:7]:
        try:
            # Try to convert each part to an integer
            keys.append(int(part))
        except ValueError:
            # If a part cannot be converted into an integer, ignore it
            keys.append(0)

    return keys

def process_trial(trial_dir):
    # Create an instance of FileStruct
    file_struct = FileStruct()

    if not os.path.isdir(trial_dir):
        return file_struct

    print(f"Processing {trial_dir}")
    trial_num = trial_dir.split('_')[-1]
    
    # Only look for .png and .wav files
    mic_files_1 = glob.glob(os.path.join(trial_dir, "mic1_audio_cmd_trim/*.wav"))
    mic_files_2 = glob.glob(os.path.join(trial_dir, "mic2_audio_cmd_trim/*.wav"))
    
    png_files_rgb = glob.glob(os.path.join(trial_dir, "rgb_image_cmd/*.png"))
    png_files_thr = glob.glob(os.path.join(trial_dir, "thr_image_cmd/*.png"))
    
    all_files = mic_files_1 + mic_files_2 + png_files_rgb + png_files_thr

    # Add the first wave file to the FileStruct
    if mic_files_1:
        file_struct.add_wave(mic_files_1[0])
    elif mic_files_2:
        file_struct.add_wave(mic_files_2[0])

    # Add all png files to the FileStruct
    for png_file in png_files_rgb + png_files_thr:
        file_struct.add_png(png_file)
    file_struct.print()
    return file_struct

def process_command_id(cmd_id, mic_files, png_files_rgb, png_files_thr, trial_dir):
    mic_files_cmd = [f for f in mic_files if re.search(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).wav$', f).group(1) == cmd_id]
    png_files_rgb_cmd = [f for f in png_files_rgb if re.search(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).png$', f).group(1) == cmd_id]
    png_files_thr_cmd = [f for f in png_files_thr if re.search(r'(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).png$', f).group(1) == cmd_id]
    if len(mic_files_cmd) == 0:
        return
    num_images_per_audio = len(png_files_rgb_cmd) // len(mic_files_cmd)
    for mic_file in mic_files_cmd:
        process_mic_file(mic_file, cmd_id, num_images_per_audio, png_files_rgb_cmd, png_files_thr_cmd, trial_dir, mic_files_cmd)

def process_mic_file(mic_file, cmd_id, num_images_per_audio, png_files_rgb_cmd, png_files_thr_cmd, trial_dir, mic_files_cmd):
    script_dir = get_script_dir()
    mic_num = os.path.splitext(os.path.basename(mic_file))[0].split('_')[-1]
    start_index = mic_files_cmd.index(mic_file) * num_images_per_audio
    end_index = start_index + num_images_per_audio

    for rgb_file in png_files_rgb_cmd[start_index:end_index]:
        output_file_rgb = os.path.join(script_dir, f"{trial_dir}_rgb_cmd{cmd_id}_mic{mic_num}_video.mkv")
        print(f"Running ffmpeg with output file {output_file_rgb}")

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('\n'.join(f"file '{x}'" for x in png_files_rgb_cmd[start_index:end_index]))
            print(f"Files to concat: {png_files_rgb_cmd[start_index:end_index]}")  # Debugging line
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', f.name, '-r', '24', '-i', mic_file, '-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '24', '-slicecrc', '1', '-c:a', 'flac', '-shortest', output_file_rgb])
            os.unlink(f.name)

    for thr_file in png_files_thr_cmd[start_index:end_index]:
        output_file_thr = os.path.join(script_dir, f"{trial_dir}_thr_cmd{cmd_id}_mic{mic_num}_video.mkv")
        print(f"Running ffmpeg with output file {output_file_thr}")

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('\n'.join(f"file '{x}'" for x in png_files_thr_cmd[start_index:end_index]))
            print(f"Files to concat: {png_files_thr_cmd[start_index:end_index]}")  # Debugging line
            subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', f.name, '-r', '24', '-i', mic_file, '-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '24', '-slicecrc', '1', '-c:a', 'flac', '-shortest', output_file_thr])
            os.unlink(f.name)

def main(root_dir):
    script_dir = get_script_dir()
    for dir in glob.glob(os.path.join(root_dir, 'sub_*_ia')):
        process_directory(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some directories.')
    parser.add_argument('root_dir', type=str, help='The root directory to process')
    args = parser.parse_args()
    main(args.root_dir)
