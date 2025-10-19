import math
import pandas as pd
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from scipy.io import loadmat
import numpy as np
import random


def read_prepared_data(args):
    data = []

    for l in range(len(args.ConType)):
        label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")
        target = []
        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, 2:]  # KUL,DTU

            data.append(eeg_data)
            target.append(label.iloc[k, args.label_col])

    return data, target


def get_data_from_mat(mat_path):
    mat_eeg_data = []
    mat_event_data = []
    matstruct_contents = loadmat(mat_path)
    matstruct_contents = matstruct_contents['data']
    mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
    mat_event_value = mat_event[0]['value']  # 1*60 1=male, 2=female
    mat_eeg = matstruct_contents[0, 0]['eeg']  # 60 trials 3200*66
    for i in range(mat_eeg.shape[1]):
        mat_eeg_data.append(mat_eeg[0, i])
        mat_event_data.append(mat_event_value[i][0][0])

    return mat_eeg_data, mat_event_data


import numpy as np


def sliding_window(eeg_datas, labels, args, out_channels):
    """
    Args:
        eeg_datas: list of np.ndarray, 每个元素 shape=(time, channels)
        labels: list of int, 每个 trial 的标签
        args: 参数对象，必须有 window_length 和 overlap
        out_channels: int, EEG通道数
    Returns:
        train_eeg, test_eeg: np.ndarray, shape=(num_windows, window_size, out_channels)
        train_label, test_label: np.ndarray, shape=(num_windows, 1)
    """
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    train_eeg, test_eeg = [], []
    train_label, test_label = [], []

    for m in range(len(labels)):
        eeg = eeg_datas[m]  # shape: (time, channels)
        label = labels[m]

        windows = []
        window_labels = []

        # 如果 trial 太短，直接补零
        if eeg.shape[0] < window_size:
            pad_len = window_size - eeg.shape[0]
            padded = np.pad(eeg, ((0, pad_len), (0, 0)), mode='constant')
            windows.append(padded)
            window_labels.append(label)
        else:
            for i in range(0, eeg.shape[0] - window_size + 1, stride):
                windows.append(eeg[i:i + window_size, :])
                window_labels.append(label)

        split_idx = int(len(windows) * 0.9)
        train_eeg.append(np.array(windows[:split_idx]))
        test_eeg.append(np.array(windows[split_idx:]))
        train_label.append(np.array(window_labels[:split_idx]).reshape(-1, 1))
        test_label.append(np.array(window_labels[split_idx:]).reshape(-1, 1))

    # 拼接所有 trial
    train_eeg = np.vstack(train_eeg).reshape(-1, window_size, out_channels)
    test_eeg = np.vstack(test_eeg).reshape(-1, window_size, out_channels)
    train_label = np.vstack(train_label)
    test_label = np.vstack(test_label)

    print(f"Train samples: {train_eeg.shape[0]}, Test samples: {test_eeg.shape[0]}")
    return train_eeg, test_eeg, train_label, test_label


def new_sliding_window(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(128 * (1 - args.overlap))

    train_eeg = []
    test_eeg = []
    train_label = []
    test_label = []

    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i + window_size, :]
            windows.append(window)
            new_label.append(label)
        train_eeg.append(np.array(windows)[:int(len(windows) * 0.9)])
        test_eeg.append(np.array(windows)[int(len(windows) * 0.9):])
        train_label.append(np.array(new_label)[:int(len(windows) * 0.9)])
        test_label.append(np.array(new_label)[int(len(windows) * 0.9):])

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, test_eeg, train_label, test_label


def sliding_window_csp(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    eeg_set = []
    label_set = []

    for m in range(len(labels)):  # labels 0-19
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i + window_size, :]
            windows.append(window)
            new_label.append(label)

        eeg_set.append(np.array(windows))
        label_set.append(np.array(new_label))

    eeg_set = np.stack(eeg_set, axis=0).reshape(-1, window_size, out_channels)
    label_set = np.stack(label_set, axis=0).reshape(-1, 1)

    return eeg_set, label_set


def within_data(eeg_datas, labels):
    train_datas = []
    test_datas = []
    train_labels = []
    test_labels = []

    for m in range(len(labels)):  # labels 0-19
        eeg = eeg_datas[m]
        label = labels[m]

        train_datas.append(np.array(eeg)[:, :int(eeg.shape[1] * 0.9)])
        test_datas.append(np.array(eeg)[:, int(eeg.shape[1] * 0.9):])
        train_labels.append(np.array(label))
        test_labels.append(np.array(label))

    train_datas = np.stack(train_datas, axis=0)
    test_datas = np.stack(test_datas, axis=0)
    train_labels = np.stack(train_labels, axis=0)
    test_labels = np.stack(test_labels, axis=0)

    return train_datas, test_datas, train_labels, test_labels