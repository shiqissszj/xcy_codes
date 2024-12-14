import os

from helper_code import is_integer


# Find patient data files.
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        # 返回指定的文件夹包含的文件或文件夹的名字的列表
        root, extension = os.path.splitext(f)
        # 分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作
        # root 存储文件名 extension存储扩展名
        if not root.startswith(".") and extension == ".txt":
            filename = os.path.join(data_folder, f)
            # 以上函数是将f拼接在data_folder 后面并且在后面的语句中加入元组中
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    for filename in filenames:
        print(os.path.split(filename)[1][:-4])

    if all(is_integer(root) for root in roots):
        filenames = sorted(
            filenames, key=lambda filename: int(os.path.split(filename)[1][:-4])
        )
        print(filenames)
    return filenames


# Find patient data files.
def find_wav_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        # 返回指定的文件夹包含的文件或文件夹的名字的列表
        root, extension = os.path.splitext(f)
        # 分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作
        # root 存储文件名 extension存储扩展名
        if not root.startswith(".") and extension == ".wav":
            filename = os.path.join(data_folder, f)
            # 以上函数是将f拼接在data_folder 后面并且在后面的语句中加入元组中
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    # for filename in filenames:
        # print(os.path.split(filename)[1][:-4])

    if all(is_integer(root) for root in roots):
        filenames = sorted(
            filenames, key=lambda filename: int(os.path.split(filename)[1][:-4])
        )
        # print(filenames)
    return filenames


# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, "r") as f:
        data = f.read()
        # 文件读操作 ，将文件数据读入data中
    return data
