# Get murmur from patient data.
# 获取文件中杂音的信息 三种 存在 缺席 未知
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from helper_code import get_num_locations, load_wav_file

def get_murmur(data):
    murmur = None
    for text in data.split("\n"):
        if text.startswith("#Murmur:"):
            murmur = text.split(": ")[1]
    if murmur is None:
        raise ValueError(
            "No murmur available. Is your code trying to load labels from the hidden data?"
        )
    return murmur


# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for text in data.split("\n"):
        if text.startswith("#Outcome:"):
            outcome = text.split(": ")[1]
    if outcome is None:
        raise ValueError(
            "No outcome available. Is your code trying to load labels from the hidden data?"
        )
    return outcome


# Get outcome from patient data.
def get_murmurmost(data):
    murmurmost = None
    for text in data.split("\n"):
        if text.startswith("#Most audible location"):
            murmurmost = text.split(": ")[1]

    # print(murmurmost)
    return murmurmost


# Get outcome from patient data.
def get_murmurlocation(data):
    murmurlocation = None
    for text in data.split("\n"):
        if text.startswith("#Murmur locations"):
            murmurlocation = text.split(": ")[1]

    # print(murmurlocation)
    return murmurlocation


def get_pitch_s(data):
    murmurlocation = None
    for text in data.split("\n"):
        if text.startswith("#Systolic murmur pitch"):
            murmurlocation = text.split(": ")[1]

    # print(murmurlocation)
    return murmurlocation


def get_s_pitch(data):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1: num_locations + 1]
    recording_information_id = data.split("\n")[0: 1]
    # print(recording_information_id) =['50078 4 4000']
    # print(recording_information_id)
    # 上述代码获取的信息是文件中后缀为.hea .tsv .wav文件
    # print(recording_information)
    entries_id = recording_information_id[0].split(" ")

    recording_file_num = entries_id[1]
    recording_file_num = int(recording_file_num)
    # print(recording_file_num)
    if recording_file_num <= 4:
        mel_specs = list()
        s_pitchs = list()
        d_pitchs = list()
        murmurclass = list()
        for i in range(num_locations):
            entries = recording_information[i].split(" ")
            recording_file = entries[2]

            # filename = os.path.join(data_directory, recording_file)
            # print(recording_file)
            recording = recording_file.split("_")
            recording = recording[1]
            recording = recording.split(".")
            recording = recording[0]
            murmurclass.append(recording)
            # print(recording)
            # print(murmurclass)
            # print("------------")
        s_pitchs = list()
        s_pitch = None
        murmurlocation = get_murmurlocation(data)
        murmurmost = get_murmurmost(data)
        murmurlocation_classes = murmurclass
        murmur = get_murmur(data)
        if murmur == "Absent":
            for i in range(recording_file_num):
                s_pitch = [0, 0, 0, 0, 1]
                s_pitchs.append(s_pitch)
        elif murmur == "Unknown":
            for i in range(recording_file_num):
                s_pitch = [0, 0, 0, 1, 0]
                s_pitchs.append(s_pitch)

        elif murmur == "Present":

            for current_murmurmost in murmurlocation_classes:
                if current_murmurmost == murmurmost:
                    if get_pitch_s(data) == "Low":
                        s_pitch = [0, 0, 1, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "Medium":
                        s_pitch = [0, 1, 0, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "High":
                        s_pitch = [1, 0, 0, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "nan":
                        s_pitch = [0, 0, 0, 0, 1]
                        s_pitchs.append(s_pitch)
                elif current_murmurmost in murmurlocation:
                    if get_pitch_s(data) == "High":
                        s_pitch = [0, 0, 1, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "Medium":
                        s_pitch = [0, 0, 1, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "Low":
                        s_pitch = [0, 0, 1, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "nan":
                        s_pitch = [0, 0, 0, 0, 1]
                        s_pitchs.append(s_pitch)
                elif not current_murmurmost in murmurlocation:
                    s_pitch = [0, 0, 0, 0, 1]
                    s_pitchs.append(s_pitch)
        s_pitchs = np.array(s_pitchs)
        s_pitchs = s_pitchs.astype(np.float64)
        # print(s_pitchs.dtype)
        # print(s_pitchs)
        return s_pitchs
    elif recording_file_num > 4:
        mel_specs = list()
        s_pitchs = list()
        d_pitchs = list()
        murmurclass = list()
        for i in range(num_locations):
            entries = recording_information[i].split(" ")
            recording_file = entries[2]

            # filename = os.path.join(data_directory, recording_file)
            # print(recording_file)
            recording = recording_file.split("_")
            recording = recording[1]
            recording = recording.split(".")
            recording = recording[0]
            murmurclass.append(recording)
            # print(recording)
            # print(murmurclass)
            # print("------------")
        s_pitchs = list()
        s_pitch = None
        murmurlocation = get_murmurlocation(data)
        murmurmost = get_murmurmost(data)
        murmurlocation_classes = murmurclass
        murmur = get_murmur(data)
        if murmur == "Absent":
            for i in range(recording_file_num):
                s_pitch = [0, 0, 0, 0, 1]
                s_pitchs.append(s_pitch)
        elif murmur == "Unknown":
            for i in range(recording_file_num):
                s_pitch = [0, 0, 0, 1, 0]
                s_pitchs.append(s_pitch)

        elif murmur == "Present":

            for current_murmurmost in murmurlocation_classes:
                if current_murmurmost == murmurmost:
                    if get_pitch_s(data) == "Low":
                        s_pitch = [0, 0, 1, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "Medium":
                        s_pitch = [0, 1, 0, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "High":
                        s_pitch = [1, 0, 0, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "nan":
                        s_pitch = [0, 0, 0, 0, 1]
                        s_pitchs.append(s_pitch)
                elif current_murmurmost in murmurlocation:
                    if get_pitch_s(data) == "High":
                        s_pitch = [0, 0, 1, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "Medium":
                        s_pitch = [0, 0, 1, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "Low":
                        s_pitch = [0, 0, 1, 0, 0]
                        s_pitchs.append(s_pitch)
                    elif get_pitch_s(data) == "nan":
                        s_pitch = [0, 0, 0, 0, 1]
                        s_pitchs.append(s_pitch)
                elif not current_murmurmost in murmurlocation:
                    s_pitch = [0, 0, 0, 0, 1]
                    s_pitchs.append(s_pitch)
        s_pitchs = np.array(s_pitchs)
        # print(s_pitchs)
        s_pitchs = s_pitchs.astype(np.float64)
        # print(s_pitchs.dtype)
        return s_pitchs



def get_pitch_d(data):
    murmurlocation = None
    for text in data.split("\n"):
        if text.startswith("#Diastolic murmur pitch"):
            murmurlocation = text.split(": ")[1]

    # print(murmurlocation)
    return murmurlocation


def get_d_pitch(data):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1: num_locations + 1]
    recording_information_id = data.split("\n")[0: 1]
    # print(recording_information_id) =['50078 4 4000']
    # print(recording_information_id)
    # 上述代码获取的信息是文件中后缀为.hea .tsv .wav文件
    # print(recording_information)
    entries_id = recording_information_id[0].split(" ")

    recording_file_num = entries_id[1]
    recording_file_num = int(recording_file_num)
    # print(recording_file_num)
    if recording_file_num <= 4:
        mel_specs = list()
        d_pitchs = list()
        murmurclass = list()
        for i in range(num_locations):
            entries = recording_information[i].split(" ")
            recording_file = entries[2]

            # filename = os.path.join(data_directory, recording_file)
            # print(recording_file)
            recording = recording_file.split("_")
            recording = recording[1]
            recording = recording.split(".")
            recording = recording[0]
            murmurclass.append(recording)
            # print(recording)
            # print(murmurclass)
            # print("------------")
        d_pitchs = list()
        d_pitch = None
        murmurlocation = get_murmurlocation(data)
        murmurmost = get_murmurmost(data)
        murmurlocation_classes = murmurclass
        murmur = get_murmur(data)
        if murmur == "Absent":
            for i in range(recording_file_num):
                d_pitch = [0, 0, 0, 0, 1]
                d_pitchs.append(d_pitch)
        elif murmur == "Unknown":
            for i in range(recording_file_num):
                d_pitch = [0, 0, 0, 1, 0]
                d_pitchs.append(d_pitch)

        elif murmur == "Present":

            for current_murmurmost in murmurlocation_classes:
                if current_murmurmost == murmurmost:
                    if get_pitch_d(data) == "Low":
                        d_pitch = [0, 0, 1, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "Medium":
                        d_pitch = [0, 1, 0, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "High":
                        d_pitch = [1, 0, 0, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "nan":
                        d_pitch = [0, 0, 0, 0, 1]
                        d_pitchs.append(d_pitch)
                elif current_murmurmost in murmurlocation:
                    if get_pitch_d(data) == "High":
                        d_pitch = [0, 0, 1, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "Medium":
                        d_pitch = [0, 0, 1, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "Low":
                        d_pitch = [0, 0, 1, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "nan":
                        d_pitch = [0, 0, 0, 0, 1]
                        d_pitchs.append(d_pitch)
                elif not current_murmurmost in murmurlocation:
                    d_pitch = [0, 0, 0, 0, 1]
                    d_pitchs.append(d_pitch)
        d_pitchs = np.array(d_pitchs)
        d_pitchs = d_pitchs.astype(np.float64)
        # print(s_pitchs)
        # print(d_pitchs.dtype)
        return d_pitchs
    elif recording_file_num > 4:

        mel_specs = list()
        d_pitchs = list()
        murmurclass = list()
        for i in range(num_locations):
            entries = recording_information[i].split(" ")
            recording_file = entries[2]

            # filename = os.path.join(data_directory, recording_file)
            # print(recording_file)
            recording = recording_file.split("_")
            recording = recording[1]
            recording = recording.split(".")
            recording = recording[0]
            murmurclass.append(recording)
            # print(recording)
            # print(murmurclass)
            # print("------------")
        d_pitchs = list()
        d_pitch = None
        murmurlocation = get_murmurlocation(data)
        murmurmost = get_murmurmost(data)
        murmurlocation_classes = murmurclass
        murmur = get_murmur(data)
        if murmur == "Absent":
            for i in range(recording_file_num):
                d_pitch = [0, 0, 0, 0, 1]
                d_pitchs.append(d_pitch)
        elif murmur == "Unknown":
            for i in range(recording_file_num):
                d_pitch = [0, 0, 0, 1, 0]
                d_pitchs.append(d_pitch)

        elif murmur == "Present":

            for current_murmurmost in murmurlocation_classes:
                if current_murmurmost == murmurmost:
                    if get_pitch_d(data) == "Low":
                        d_pitch = [0, 0, 1, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "Medium":
                        d_pitch = [0, 1, 0, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "High":
                        d_pitch = [1, 0, 0, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "nan":
                        d_pitch = [0, 0, 0, 0, 1]
                        d_pitchs.append(d_pitch)
                elif current_murmurmost in murmurlocation:
                    if get_pitch_d(data) == "High":
                        d_pitch = [0, 0, 1, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "Medium":
                        d_pitch = [0, 0, 1, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "Low":
                        d_pitch = [0, 0, 1, 0, 0]
                        d_pitchs.append(d_pitch)
                    elif get_pitch_d(data) == "nan":
                        d_pitch = [0, 0, 0, 0, 1]
                        d_pitchs.append(d_pitch)
                elif not current_murmurmost in murmurlocation:
                    d_pitch = [0, 0, 0, 0, 1]
                    d_pitchs.append(d_pitch)
        d_pitchs = np.array(d_pitchs)
        d_pitchs = d_pitchs.astype(np.float64)
        # print(s_pitchs)
        # print(d_pitchs.dtype) float 64
        return d_pitchs
