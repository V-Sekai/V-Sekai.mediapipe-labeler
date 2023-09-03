#!/usr/bin/env python3
import os
import glob
import subprocess
import argparse
import re
import tempfile
import numpy as np
import shutil
import locale
from natsort import natsorted

locale.setlocale(locale.LC_ALL, '')

from operator import itemgetter

class FileStruct:
    def __init__(self):
        self.wave_file = None
        self.png_files = []
        self.command_id = None  # New attribute to store the command id

    def add_wave(self, wave_file):
        cmd_id = os.path.splitext(os.path.basename(wave_file))[0].split('_')[4]  # Extract the command id from the filename
        if self.command_id is not None and self.command_id != cmd_id:
            print(f"Error: Attempted to add wave file with different command id ({cmd_id}) to FileStruct with command id {self.command_id}")
            self.print()
            return
        self.command_id = cmd_id  # Store the command id
        self.wave_file = wave_file

    def add_png(self, png_file):
        cmd_id = os.path.splitext(os.path.basename(png_file))[0].split('_')[4]  # Extract the command id from the filename
        if self.command_id != cmd_id:
            print(f"Error: Attempted to add PNG file with different command id ({cmd_id}) to FileStruct with command id {self.command_id}")
            print(f"PNG file: {png_file}")
            self.print()
            return
        if not any(file[0] == png_file for file in self.png_files):
            self.png_files.append((png_file, self.wave_file))

    def print(self):
        print(f"Wave file: {self.wave_file}")
        print("PNG files:")
        for png_tuple in self.png_files:
            print(f"PNG file: {png_tuple[0]}, Wave file: {png_tuple[1]}")


def main(root_dir):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Create a list to store FileStruct instances
    file_structs = []

    last_cmd_number = None  # Store the last command number

    for dir in glob.glob(os.path.join(root_dir, 'sub_*_ia')):
        if not os.path.isdir(dir):
            continue
        print(f"Processing {dir}")
        print("Directory contents before processing:")
        print(os.listdir(dir))

        for trial_dir in glob.glob(os.path.join(dir, 'trial_*')):
            if not os.path.isdir(trial_dir):
                continue

            print(f"Processing {trial_dir}")
            trial_num = trial_dir.split('_')[-1]

            mic_files_1 = natsorted(glob.glob(os.path.join(trial_dir, "mic1_audio_cmd_trim/*.wav")))
            mic_files_2 = natsorted(glob.glob(os.path.join(trial_dir, "mic2_audio_cmd_trim/*.wav")))

            png_files_rgb = natsorted(glob.glob(os.path.join(trial_dir, "rgb_image_cmd_aligned/*.png")))
            png_files_thr = natsorted(glob.glob(os.path.join(trial_dir, "thr_image_cmd_aligned/*.png")))
            all_files = mic_files_1 + mic_files_2 + png_files_rgb + png_files_thr
            
            # Initialize the first FileStruct
            file_struct_rgb = FileStruct()
            file_struct_thr = FileStruct()

            # Add the first wave file to the FileStruct
            if mic_files_1:
                file_struct_rgb.add_wave(mic_files_1[0])
                file_struct_thr.add_wave(mic_files_1[0])
            elif mic_files_2:
                file_struct_rgb.add_wave(mic_files_2[0])
                file_struct_thr.add_wave(mic_files_2[0])

            # Add all png files to the FileStruct
            for png_file in png_files_rgb:
                # Extract the command number from the filename
                cmd_number = int(png_file.split('_')[-4])

                # If the command number is different from the last one, create a new FileStruct
                if cmd_number != last_cmd_number:
                    file_struct_rgb = FileStruct()
                    file_struct_thr = FileStruct()
                    file_struct_rgb.add_wave(mic_files_1[0] if mic_files_1 else mic_files_2[0])
                    file_struct_thr.add_wave(mic_files_1[0] if mic_files_1 else mic_files_2[0])
                    file_structs.append(file_struct_rgb)
                    file_structs.append(file_struct_thr)

                file_struct_rgb.add_png(png_file)
                last_cmd_number = cmd_number  # Update the last command number

            for png_file in png_files_thr:
                file_struct_thr.add_png(png_file)

            file_struct_rgb.print()
            file_struct_thr.print()

            # Add the processed FileStruct to the list
            file_structs.append(file_struct_rgb)
            file_structs.append(file_struct_thr)

    # Call process_mic_files function here
    for file_struct in file_structs:
        mic_file = file_struct.wave_file

        # Raise an exception if mic_file is None
        if mic_file is None:
            raise ValueError("No wave file found")

        cmd_id, mic_num = os.path.splitext(os.path.basename(mic_file))[0].split('_')[:2]
        num_images_per_audio = len(file_struct.png_files)

        # Create a dictionary to store png files by their prefix
        png_files_by_prefix = {}

        for i in range(num_images_per_audio):
            png_file, _ = file_struct.png_files[i]  # Access the first element of the tuple
            prefix = os.path.splitext(os.path.basename(png_file))[0].split('_')[0]

            if prefix not in png_files_by_prefix:
                png_files_by_prefix[prefix] = []

            png_files_by_prefix[prefix].append(png_file)

        for prefix, png_files in png_files_by_prefix.items():
            output_file = os.path.join(script_dir, f"{cmd_id}_mic{mic_num}_video_{prefix}.mkv")
            print(f"Running ffmpeg with output file {output_file}")

            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy all png files to the temporary directory
                for png_file in png_files:
                    shutil.copy2(png_file, temp_dir)

                # Run ffmpeg on the copied png files
                subprocess.run(['ffmpeg', '-y', '-pattern_type', 'glob', '-i', os.path.join(temp_dir, '*.png'), '-r', '24', '-i', mic_file, '-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '24', '-slicecrc', '1', '-c:a', 'flac', '-shortest', output_file])
        file_struct.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some directories.')
    parser.add_argument('root_dir', type=str, help='The root directory to process')
    args = parser.parse_args()
    main(args.root_dir)
