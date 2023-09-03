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

file_structs = []
class FileStruct:
    def __init__(self):
        self.wave_file = None
        self.png_files = []
        self.command_id = None  # Existing attribute to store the command id
        self.sub_id = None
        self.trial_id = None
        self.session_id = None
        self.pos_id = None
        self.frame_id = None
        self.stream_id = None
        self.mic_id = None
        self.mic_num = None 

    def add_wave(self, wave_file):
        file_parts = os.path.splitext(os.path.basename(wave_file))[0].split('_')
        sub_id = int(file_parts[0])
        trial_id = int(file_parts[1])
        session_id = int(file_parts[2])
        pos_id = int(file_parts[3])
        command_id = int(file_parts[4])
        version = int(file_parts[5])  # Extract version

        if self.command_id is not None and (self.sub_id != sub_id or self.trial_id != trial_id or self.session_id != session_id or self.pos_id != pos_id or self.command_id != command_id):
            print(f"Error: Attempted to add wave file with different attributes to FileStruct")
            self.print()
            return

        self.sub_id = sub_id
        self.trial_id = trial_id
        self.session_id = session_id
        self.pos_id = pos_id
        self.command_id = command_id
        self.version = version
        self.wave_file = wave_file  # Add this line to actually store the wave file path

    def add_png(self, png_file, file_structs):  # Added file_structs as an argument
        file_parts = os.path.splitext(os.path.basename(png_file))[0].split('_')
        sub_id = int(file_parts[0])
        trial_id = int(file_parts[1])
        session_id = int(file_parts[2])
        pos_id = int(file_parts[3])
        command_id = int(file_parts[4])
        frame_id = int(file_parts[5])  # Add this line
        version = int(file_parts[6])  # Add this line


        # Find a FileStruct with matching attributes or create a new one
        file_struct = next((fs for fs in file_structs if fs.sub_id == sub_id and fs.trial_id == trial_id and fs.session_id == session_id and fs.pos_id == pos_id and fs.command_id == command_id and fs.frame_id == frame_id and fs.version == version), None)  # Add frame_id and version to the comparison
        if file_struct is None:
            print(f"No matching FileStruct found for PNG file {png_file}")
            file_struct = FileStruct()
            file_struct.sub_id = sub_id
            file_struct.trial_id = trial_id
            file_struct.session_id = session_id
            file_struct.pos_id = pos_id
            file_struct.command_id = command_id
            file_struct.frame_id = frame_id
            file_struct.version = version
            file_structs.append(file_struct)

        if file_struct is not None:
            file_struct.png_files.append(png_file)

    def print(self):
        print(f"Wave file: {self.wave_file}")
        print(f"PNG files: {self.png_files}")
        print(f"Command ID: {self.command_id}")
        print(f"Sub ID: {self.sub_id}")
        print(f"Trial ID: {self.trial_id}")
        print(f"Session ID: {self.session_id}")
        print(f"Pos ID: {self.pos_id}")
        print(f"Frame ID: {self.frame_id}")
        print(f"Stream ID: {self.stream_id}")
        print(f"Mic ID: {self.mic_id}")
        print(f"Mic Num: {self.mic_num}")

def main(root_dir):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Processing .wav files
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

            # Create a dictionary that maps command IDs and versions to .wav files
            wave_files_by_cmd_id_version = {os.path.splitext(os.path.basename(wave_file))[0].split('_')[4] + '_' + os.path.splitext(os.path.basename(wave_file))[0].split('_')[5]: wave_file for wave_file in mic_files_1}

            # Initialize the first FileStruct
            file_struct_rgb = FileStruct()

            # Add wave files to the FileStruct
            for wave_file in mic_files_1:
                file_parts = os.path.splitext(os.path.basename(wave_file))[0].split('_')

                if len(file_parts) > 5:  # Check if there are enough parts before accessing them
                    cmd_number_version = file_parts[4] + '_' + file_parts[5]  # Use index 5 instead of 6
                    file_struct_rgb = FileStruct()  # Initialize FileStruct here
                    file_struct_rgb.add_wave(wave_file)
                    file_struct_rgb.mic_num = 1  # Set mic_num to 1 for mic_files_1
                    file_structs.append(file_struct_rgb)
                else:
                    print(f"Skipping {wave_file} due to insufficient parts")

    for dir in glob.glob(os.path.join(root_dir, 'sub_*_ia')):
        if not os.path.isdir(dir):
            continue

        for trial_dir in glob.glob(os.path.join(dir, 'trial_*')):
            # Check if trial_dir is actually a directory before listing its contents
            if os.path.isdir(trial_dir):
                png_files = glob.glob(os.path.join(trial_dir, 'rgb_image_cmd_aligned/*.png'), recursive=True)

                if not png_files:
                    print(f"No PNG files found in directory {trial_dir}")

                for png_file in png_files:
                    # Extract attributes from the png file
                    file_parts = os.path.splitext(os.path.basename(png_file))[0].split('_')
                    sub_id = int(file_parts[0])
                    trial_id = int(file_parts[1])
                    session_id = int(file_parts[2])
                    pos_id = int(file_parts[3])
                    command_id = int(file_parts[4])

                    file_struct = next((fs for fs in file_structs if fs.sub_id == sub_id and fs.trial_id == trial_id and fs.session_id == session_id and fs.pos_id == pos_id and fs.command_id == command_id), None)
                    
                    if file_struct is not None:
                        # Add the png file to the found FileStruct
                        file_struct.png_files.append(png_file)
                    else:
                        print(f"No matching FileStruct found for PNG file {png_file}")


                # Combine wav with png using ffmpeg
                for file_struct in file_structs:
                    output_file = os.path.join(script_dir, f"{file_struct.sub_id}_{file_struct.trial_id}_{file_struct.session_id}_{file_struct.pos_id}_{file_struct.command_id}_{file_struct.version}_video.mkv")
                    print(f"Running ffmpeg with output file {output_file}")
                    
                    # Create a temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Copy all png files to the temporary directory
                        for png_file in natsorted(file_struct.png_files):  # Sort the PNG files here
                            shutil.copy2(png_file, temp_dir)
                        
                        # Run ffmpeg on the copied png files
                        subprocess.run(['ffmpeg', '-y', '-pattern_type', 'glob', '-i', os.path.join(temp_dir, '*.png'), '-r', '28', '-i', file_struct.wave_file, '-c:v', 'ffv1', '-level', '3', '-g', '1', '-slices', '24', '-slicecrc', '1', '-c:a', 'flac', '-shortest', output_file])
                    
                    file_struct.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some directories.')
    parser.add_argument('root_dir', type=str, help='The root directory to process')
    args = parser.parse_args()
    main(args.root_dir)
