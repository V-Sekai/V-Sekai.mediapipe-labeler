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

locale.setlocale(locale.LC_ALL, '')

from operator import itemgetter
class FileStruct:
    def __init__(self):
        self.wave_file = None
        self.png_files = []

    def add_wave(self, wave_file):
        self.wave_file = wave_file

    def add_png(self, png_file):
        if not any(file[0] == png_file for file in self.png_files):
            self.png_files.append((png_file, self.wave_file))
            
    def sort_png_files(self):
        self.png_files.sort(key=itemgetter(0))  # Sort by PNG file name

    def print(self):
        print(f"Wave file: {self.wave_file}")
        print("PNG files:")
        for png_tuple in self.png_files:
            print(f"PNG file: {png_tuple[0]}, Wave file: {png_tuple[1]}")

def sort_key_func(item):
    # Split the filename into parts
    parts = re.split(r'(\d+)', item)

    # Convert numeric parts to integers
    parts[1::2] = map(int, parts[1::2])

    # Return a tuple of parts
    return tuple(parts)

def main(root_dir):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Create a list to store FileStruct instances
    file_structs = []

    for dir in glob.glob(os.path.join(root_dir, 'sub_*_ia')):
        if not os.path.isdir(dir):
            continue
        print(f"Processing {dir}")
        print("Directory contents before processing:")
        print(os.listdir(dir))

        for trial_dir in glob.glob(os.path.join(dir, 'trial_*')):
            # Process each trial directory and get the filled FileStruct
            file_struct_rgb = FileStruct()
            file_struct_thr = FileStruct()

            if not os.path.isdir(trial_dir):
                continue

            print(f"Processing {trial_dir}")
            trial_num = trial_dir.split('_')[-1]

            mic_files_1 = sorted(glob.glob(os.path.join(trial_dir, "mic1_audio_cmd_trim/*.wav")), key=sort_key_func)
            mic_files_2 = sorted(glob.glob(os.path.join(trial_dir, "mic2_audio_cmd_trim/*.wav")), key=sort_key_func)

            png_files_rgb = sorted(glob.glob(os.path.join(trial_dir, "rgb_image_cmd_aligned/*.png")), key=sort_key_func)
            png_files_thr = sorted(glob.glob(os.path.join(trial_dir, "thr_image_cmd_aligned/*.png")), key=sort_key_func)
            all_files = mic_files_1 + mic_files_2 + png_files_rgb + png_files_thr

            # Add the first wave file to the FileStruct
            if mic_files_1:
                file_struct_rgb.add_wave(mic_files_1[0])
                file_struct_thr.add_wave(mic_files_1[0])
            elif mic_files_2:
                file_struct_rgb.add_wave(mic_files_2[0])
                file_struct_thr.add_wave(mic_files_2[0])

            # Add all png files to the FileStruct
            for png_file in png_files_rgb:
                file_struct_rgb.add_png(png_file)
            for png_file in png_files_thr:
                file_struct_thr.add_png(png_file)

            file_struct_rgb.sort_png_files()
            file_struct_rgb.print()
            file_struct_thr.sort_png_files()
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
