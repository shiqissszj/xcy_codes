import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from find_and_load_patient_files import (
    find_patient_files,
    load_patient_data,
)
from DataProcessing.helper_code import get_num_locations, load_wav_file
from DataProcessing.label_extraction import get_murmur, get_outcome, get_murmurmost, get_murmurlocation, get_s_pitch, get_d_pitch
from DataProcessing.compute_LogMelSpecs import waveform_to_examples
from DataProcessing.deal import padding


# 定义了加载特征的函数
def net_feature_loader(
        data_directory
):

    if data_directory:
        spectrograms_train, id_train, s_pitchs_train, d_pitchs_train, murmurs_train, outcomes_train = calc_patient_features(
            data_directory
        )
        # print(murmurs_train)
        repeats = torch.zeros((len(spectrograms_train),))
        locations = torch.zeros((len(spectrograms_train),))
        # 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
        # 2054 print(len(spectrograms_train))
        
        
        for i in range(len(spectrograms_train)):
            locations[i] += len(spectrograms_train[i])
            for j in range(len(spectrograms_train[i])):
                repeats[i] += len(spectrograms_train[i][j])

          
        # print(repeats)
        # print("---------")
        murmurs_train = torch.repeat_interleave(
             torch.Tensor(np.array(murmurs_train)), repeats.to(torch.int32), dim=0
         )

         
         
        # print(murmurs_train.shape)
        # 重复张量的元素
        # 输入参数：
        #
        # input (类型：torch.Tensor)：输入张量
        # repeats（类型：int或torch.Tensor）：每个元素的重复次数。repeats参数会被广播来适应输入张量的维度
        # dim（类型：int）需要重复的维度。默认情况下，将把输入张量展平（flatten）为向量，然后将每个元素重复repeats次，并返回重复后的张量。
        # print(murmurs_train)
        # outcomes_train = torch.repeat_interleave(
        #     torch.Tensor(np.array(outcomes_train)), repeats.to(torch.int32), dim=0
        # )
        # print(s_pitchs_train)
        # print("------------------")
        # s_pitchs_train = [val.numpy() for val in s_pitchs_train ]
        # s_pitchs_train =np.array(s_pitchs_train)
        # print(s_pitchs_train.dtype)
        tmp_l = []
        for i in range(len(s_pitchs_train)):
            p = list(s_pitchs_train[i])
            tmp_l.extend(p)

        tmp_l = np.array(tmp_l)
        s_pitchs_train = torch.Tensor(tmp_l)
        # print(s_pitchs_train.shape)

        repeats = torch.zeros((len(s_pitchs_train),))
        # 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
        # 2054 print(len(spectrograms_train))
        pos=0
        for i in range(len(spectrograms_train)):
            for j in range(len(spectrograms_train[i])):
                repeats[pos] = len(spectrograms_train[i][j])
                pos = pos + 1
        # print(repeats)
        # print("---------")
        s_pitchs_train = torch.repeat_interleave(
             s_pitchs_train, repeats.to(torch.int32), dim=0
         )
        '''
        s_pitchs_train = padding(s_pitchs_train, id_train)
        s_pitchs_train = torch.tensor(s_pitchs_train, requires_grad=True)[
                         :, None, :, :
                         ].float()
        s_pitchs_train = torch.repeat_interleave(
             s_pitchs_train, repeats.to(torch.int32), dim=0
         )
         '''
        # print(s_pitchs_train)
        # s_pitchs_train = torch.Tensor(np.array(s_pitchs_train))
        
        tmp_l = []
        for i in range(len(d_pitchs_train)):
            p = list(d_pitchs_train[i])
            tmp_l.extend(p)

        tmp_l = np.array(tmp_l)
        d_pitchs_train = torch.Tensor(tmp_l)
        # print(d_pitchs_train.shape)
        d_pitchs_train = torch.repeat_interleave(
             d_pitchs_train, repeats.to(torch.int32), dim=0
        )
        '''
        d_pitchs_train = padding(d_pitchs_train, id_train)
        d_pitchs_train = torch.tensor(d_pitchs_train, requires_grad=True)[
                         :, None, :, :
                         ].float()
        d_pitchs_train = torch.repeat_interleave(
             d_pitchs_train, repeats.to(torch.int32), dim=0
         )
        # d_pitchs_train = torch.Tensor(np.array(d_pitchs_train))
        '''
        # print3
        # 下列函数在给定维度上对输入的张量序列seq 进行连接操作


    return (
        spectrograms_train,
        murmurs_train,
        outcomes_train,
        s_pitchs_train,
        d_pitchs_train,
        locations 
    )


def patient_feature_loader(recalc_features, data_directory, output_directory):
    if recalc_features == "True":
        spectrograms, murmurs, outcomes = calc_patient_features(data_directory)
        with open(output_directory + "spectrograms", "wb") as fp:
            pickle.dump(spectrograms, fp)
        with open(output_directory + "murmurs", "wb") as fp:
            pickle.dump(murmurs, fp)
        with open(output_directory + "outcomes", "wb") as fp:
            pickle.dump(outcomes, fp)
    else:
        with open(output_directory + "spectrograms", "rb") as fp:
            spectrograms = pickle.load(fp)
        with open(output_directory + "murmurs", "rb") as fp:
            murmurs = pickle.load(fp)
        with open(output_directory + "outcomes", "rb") as fp:
            outcomes = pickle.load(fp)

    return spectrograms, murmurs, outcomes


# Load recordings. 注意这里的data变量存储的只是文件中txt文件的内容
def load_spectrograms(data_directory, data, current_murmurlocation, current_murmurmost, s_pitch, d_pitch):
    num_locations = get_num_locations(data)
    recording_information = data.split("\n")[1: num_locations + 1]
    # print(recording_information)
    # 上述代码获取的信息是文件中后缀为.hea .tsv .wav文件
    # print(recording_information)
    mel_specs = list()
    s_pitchs = list()
    d_pitchs = list()
    for i in range(num_locations):
        entries = recording_information[i].split(" ")
        recording_file = entries[2]
        filename = os.path.join(data_directory, recording_file)
        # print(recording_file)
        recording, frequency = load_wav_file(filename)
        recording = recording / 32768
        mel_spec = waveform_to_examples(recording, frequency)
        # print(mel_spec)
        # print("------------------------")
        mel_specs.append(mel_spec)
        
        # print(mel_spec)
    # print(len(mel_specs))
    return mel_specs, s_pitchs, d_pitchs


# 这里返回的是log mel 数据

#        [[[-1.2907, -1.5411, -2.0187,  ..., -4.4008, -4.4587, -4.5489],
#         [-1.8707, -2.1128, -2.9819,  ..., -4.5142, -4.4946, -4.4962],
#          [-2.2349, -2.4686, -2.3867,  ..., -4.4531, -4.5307, -4.5450],
#         ...,
#        [-2.6547, -2.8744, -2.8227,  ..., -4.3185, -4.4027, -4.4770],
#       [-2.4191, -2.6473, -2.9253,  ..., -4.4074, -4.4944, -4.5524],
#      [-2.2604, -2.4934, -2.5226,  ..., -4.5013, -4.4858, -4.5184]]]],
#  grad_fn=<ToCopyBackward0>), tensor([[[[-0.8423, -1.0967, -0.3802,  ..., -3.1717, -3.4216, -3.5896],
#     [-0.7072, -0.9624, -1.3446,  ..., -2.5202, -3.4222, -3.9881],
#    [-0.8306, -1.0850, -1.2454,  ..., -3.1196, -3.4497, -4.2798],
#   ...,
#  [-2.3145, -2.5460, -2.7071,  ..., -4.3045, -4.3007, -4.4245],
# [-3.5867, -3.7455, -3.1314,  ..., -4.3712, -4.3531, -4.3839],
# [-2.4863, -2.7122, -2.6569,  ..., -4.4524, -4.5120, -4.5262]]],

# 以下函数主要是从目标文件夹中获取 数据信息，并且将这些信息通过数组存储进程序中

def calc_patient_features(data_directory):
    murmurmost_classes = ["AV", "MV", "PV", "TV"]
    num_murmurmost_classes = len(murmurmost_classes)
    murmurlocation_classes = ["AV", "MV", "PV", "TV"]
    num_murmurlocation_classes = len(murmurlocation_classes)
    # 新奇一列四类的

    murmur_classes = ["Present", "Unknown", "Absent"]

    num_murmur_classes = len(murmur_classes)
    outcome_classes = ["Abnormal", "Normal"]
    num_outcome_classes = len(outcome_classes)
    s_pitch_classes = ["Absent", "Unknown", "high", "medium", "low"]
    d_pitch_classes = ["Absent", "Unknown", "high", "medium", "low"]
    num_s_pitch_classes = len(s_pitch_classes)
    num_d_pitch_classes = len(d_pitch_classes)
    s_pitchs = list()
    d_pitchs = list()
    # Find the patient data files.
    patient_files = find_patient_files(data_directory)
    num_patient_files = len(patient_files)
    spectrograms = list()
    murmurs = list()
    outcomes = list()
    murmurmosts = list()
    murmurlocations = list()
    ids = list()
    pitchs = list()
    id = 0
    # num_patient_files
    for i in tqdm(range(num_patient_files)):

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        # print(current_patient_data)

        current_murmurlocation = np.zeros(num_murmurmost_classes, dtype=int)
        murmurlocation = get_murmurlocation(current_patient_data)
        murmurlocation = murmurlocation.split("+")
        # print(murmurlocation)
        murmurlocation = np.vstack(murmurlocation)
        for k in range(len(murmurlocation)):
            murmurlocation1 = murmurlocation[k]
            if murmurlocation1 in murmurlocation_classes:
                # 如果杂音的信息存在在现有的三个中
                j = murmurlocation_classes.index(murmurlocation1)
                # print(j) # 这里j存储的是murmur在数组中的序号 0 为"Present", 1 "Unknown", 2 "Absent"
                current_murmurlocation[j] = 1
        murmurlocations.append(current_murmurlocation)
        # print(murmurlocations)
        # 上述代码是自己添加的一个murmurlocation数组初始化，比如存在pv tv 那么 该序号上的数存为1 不存在则为0

        current_murmurmost = np.zeros(num_murmurmost_classes, dtype=int)
        murmurmost = get_murmurmost(current_patient_data)
        if murmurmost in murmurmost_classes:
            # 如果杂音的信息存在在现有的三个中
            j = murmurmost_classes.index(murmurmost)
            # print(j) # 这里j存储的是murmur在数组中的序号 0 为"Present", 1 "Unknown", 2 "Absent"
            current_murmurmost[j] = 1
        murmurmosts.append(current_murmurmost)
        # print(murmurmosts)
        # 在这里将杂音最多的位置存储在murmurmosts数组中，那个位置为1那个位置就是最多的
        # !!!!!!!!!!!!!!
        current_s_pitch = np.zeros(num_s_pitch_classes, dtype=int)
        s_pitch = get_s_pitch(current_patient_data)
        s_pitch = np.array(s_pitch)
        s_pitch = s_pitch.astype(np.float64)

        # print(s_pitch)
        d_pitch = get_d_pitch(current_patient_data)
        d_pitch = np.array(d_pitch)
        d_pitch = d_pitch.astype(np.float64)
        # print(d_pitch)
        current_spectrograms, current_s_pitch, current_d_pitch = load_spectrograms(data_directory, current_patient_data,
                                                                                   current_murmurlocation,
                                                                                   current_murmurmost, s_pitch, d_pitch)
        current_s_pitch = get_s_pitch(current_patient_data)
        current_d_pitch = get_d_pitch(current_patient_data)
        spectrograms.append(current_spectrograms)
        
        # spectrograms.extend(current_spectrograms)
        # print(spectrograms)
        # print(current_spectrograms)
        # s_pitchs.extend(s_pitch)
        # d_pitchs.extend(d_pitch)
        s_pitchs.append(s_pitch)
        d_pitchs.append(d_pitch)

        id = id + 1
        # print(s_pitchs)
        pitchs.extend(current_s_pitch)
        pitchs.extend(current_d_pitch)
        l = len(current_spectrograms)
        for j in range(l):
            ids.append(i)
        # print(ids)

        current_murmur = np.zeros(num_murmur_classes, dtype=int)
        murmur = get_murmur(current_patient_data)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
        murmurs.append(current_murmur)
        # Outcome
        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)
        if murmur in murmur_classes:
            j = murmur_classes.index(murmur)
            current_murmur[j] = 1
    # print(spectrograms)
    # print(s_pitchs.dtype)
    # print(s_pitchs)
    # s_pitchs = np.array(s_pitchs)
    for i in range(len(s_pitchs)):
        s_pitchs[i] = s_pitchs[i].astype(np.float64)
        # print(s_pitchs[i].dtype)
    # print(s_pitchs)
    # print(s_pitchs.dtype)
#    d_pitchs = np.array(d_pitchs)
    # print(s_pitchs)
    # print(id)
    # print(len(spectrograms))
    return spectrograms, id, s_pitchs, d_pitchs, murmurs, outcomes
