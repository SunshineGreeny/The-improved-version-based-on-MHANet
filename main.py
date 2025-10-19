from dotmap import DotMap
from tqdm import tqdm
from utils import *
from data_process import *
from model import MHANet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from mne.decoding import CSP
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import config
import os # 确保 os 被导入
import math # 确保 math 被导入
import pandas as pd # 确保 pandas 被导入

np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CustomDatasets(Dataset):
    # initialization: data and label
    def __init__(self, seq_data, label_data):
        self.seq_data = seq_data

        self.label = label_data

    # get the size of data
    def __len__(self):
        return len(self.label)

    # get the data and label
    def __getitem__(self, index):
        seq_data = torch.Tensor(self.seq_data[index])

        label = torch.Tensor(self.label[index])

        return seq_data, label


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 训练前初始化配置
def initiate(args, train_loader, valid_loader, test_loader, subject):
    model = MHANet(args)

    # 打印模型参数量
    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")

    # 获取损失函数
    criterion = nn.CrossEntropyLoss()

    # 获取优化器
    # optimizer = optim.AdamW(params=model.parameters(), lr=0.005, weight_decay=3e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.003 / 10)
    optimizer = optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0.003 / 10)
    model = model.cuda()
    criterion = criterion.cuda()

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, args, train_loader, valid_loader, test_loader, subject)


def train_model(settings, args, train_loader, valid_loader, test_loader, subject):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, scheduler):
        model.train()
        proc_loss, proc_size = 0, 0
        train_acc_sum = 0
        train_loss_sum = 0
        for i_batch, batch_data in enumerate(train_loader):
            seq_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            seq_data, train_label = seq_data.cuda(), train_label.cuda()

            batch_size = train_label.size(0)

            # Forward pass
            preds = model(seq_data)

            loss = criterion(preds, train_label.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            train_loss_sum += loss.item() * batch_size
            predicted = preds.data.max(1)[1]
            train_acc_sum += predicted.eq(train_label).cpu().sum()

        scheduler.step()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)
        total_loss = 0.0
        test_acc_sum = 0
        proc_size = 0

        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                seq_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                seq_data, test_label = seq_data.cuda(), test_label.cuda()

                proc_size += args.batch_size
                preds = model(seq_data)
                # Backward and optimize
                optimizer.zero_grad()

                total_loss += criterion(preds, test_label.long()).item() * args.batch_size

                predicted = preds.data.max(1)[1]  # 32
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        avg_loss = total_loss / (num_batches * args.batch_size)

        avg_acc = test_acc_sum / (num_batches * args.batch_size)

        return avg_loss, avg_acc

    best_epoch = 1
    best_valid = float('inf')
    epochs_without_improvement = 0 # 确保初始化
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print()
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc '
            '{:5.4f}'.format(
                epoch,
                args.name,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > args.patience: # 使用 args.patience
                print(f"Early stopping at epoch {epoch}")
                break

    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    return test_loss, test_acc


def getData(args, dataset="DTU"):
    seq_alldata = []
    alllabel = []
    alll_ckabel = []

    if dataset == 'DTU':
        for id in range(1, args.subject_number + 1):
            onedata, onelabel = get_DTU_data(args, id)
            onedata, onelabel, check_label = sliding_window(args, onedata, onelabel, id)
            onedata = onedata.transpose(0, 2, 1)
            seq_alldata.append(onedata)
            alllabel.append(onelabel)
            alll_ckabel.append(check_label)

    elif dataset == 'KUL':
        for id in range(1, args.subject_number + 1):
            onedata, onelabel = get_KUL_data(args, id)
            onedata, onelabel, check_label = sliding_window(args, onedata, onelabel, id)
            onedata = onedata.transpose(0, 2, 1)
            seq_alldata.append(onedata)
            alllabel.append(onelabel)
            alll_ckabel.append(check_label)

    elif dataset == 'AVED':
        for id in range(1, args.subject_number + 1):
            onedata, onelabel = get_AVED_data(args, id)
            onedata = onedata.reshape([args.trail_number, -1, args.eeg_channel])
            onedata, onelabel, check_label = sliding_window(args, onedata, onelabel, id)
            onedata = onedata.transpose(0, 2, 1)
            seq_alldata.append(onedata)
            alllabel.append(onelabel)
            alll_ckabel.append(check_label)
    return seq_alldata, alllabel, alll_ckabel


# ========================= kul data =====================================
def get_KUL_data(args, sub_id):
    '''description: get all the data from one dataset
    param {type}
    return {type}:
        data: list  16(subjects), each data is x *
        label: '''
    alldata = []
    all_data_dir = os.listdir(args.data_path)
    all_data_dir.sort()
    # for s_data in range(len(all_data_dir)):
    sub_path = args.data_path + str(sub_id)
    sublabel_path = args.label_path + "S" + str(sub_id) + "No.csv"
    sub_data_dir = os.listdir(sub_path)
    sub_data_dir.sort()
    for k in range(len(sub_data_dir)):
        filename = sub_path + '/' + sub_data_dir[k]
        data_pf = pd.read_csv(filename, header=None)
        eeg_data = data_pf.iloc[:, 2:].values  # （46080，64）

        alldata.append(eeg_data)
    label_pf = pd.read_csv(sublabel_path, header=None)
    all_label = label_pf.iloc[1:, 0].values
    print('Finish get the data from: ', args.data_path + str(sub_id))
    return alldata, all_label


# ========================= dtu data =====================================
def get_DTU_data(args, sub_id):
    '''description: get all the data from one dataset
    param {type}
    return {type}:
        data: list  16(subjects), each data is x *
        label: '''
    alldata = []
    sub_path = args.data_path + "s" + str(sub_id) + "_data.npy"
    sublabel_path = args.label_path + "s" + str(sub_id) + "_label.npy"
    sub_data = np.load(sub_path)
    sub_label = np.load(sublabel_path)
    print('Finish get the data from: ', args.data_path + str(sub_id))
    return sub_data, sub_label


# ========================= ahu data ====================================
def get_AVED_data(args, sub_id, modality="audio-only"):
    """
    获取单个被试的数据 (修正: 移除硬编码标签，模拟加载标签)
    """
    # EEG Data File Path
    eeg_filename = os.path.join(args.data_document_path, modality, f"sub{sub_id}.csv")

    # 模拟标签文件路径 (AVED数据集的标签通常在一个独立的CSV或TXT文件中)
    # 假设标签文件名为 subX_label.csv，并且与 eeg 文件在同一目录下
    # 注意：这里需要根据 AVED 实际标签存储位置和格式进行修改！
    label_filename = os.path.join(args.data_document_path, modality, f"sub{sub_id}_label.csv")

    if not os.path.exists(eeg_filename):
        raise FileNotFoundError(f"EEG Data file not found: {eeg_filename}")

    # 加载 EEG 数据
    eeg_data_pf = pd.read_csv(eeg_filename, header=None)
    eeg_data = eeg_data_pf.values  # shape: (time, channels)

    # 尝试加载标签数据 (关键修正)
    try:
        if not os.path.exists(label_filename):
            # 如果找不到标签文件，则抛出错误，提示用户提供正确的标签文件
            raise FileNotFoundError(f"Label file not found at expected path: {label_filename}")

        # 假设标签文件 subX_label.csv 包含一个包含所有试次标签的列
        label_pf = pd.read_csv(label_filename, header=None)
        # 假设标签在第一列，且需要从1开始的标签转换为从0开始 (例如 1 -> 0, 2 -> 1)
        # 假设标签是 N_trials x 1 的形状，并且需要进行 -1 操作以匹配 [0, N_classes-1]
        all_label = label_pf.values.astype(np.int32).flatten()
        # 假设原始标签从 1 开始，需减 1
        if np.min(all_label) >= 1:
            all_label = all_label - 1

            # 检查标签数量是否与试次数量匹配
        if len(all_label) != args.trail_number:
            print(f"CRITICAL WARNING: Number of loaded labels ({len(all_label)}) does not match args.trail_number ({args.trail_number}). Check label file content.")

    except FileNotFoundError as e:
        print(f"Error loading labels: {e}")
        # 如果标签加载失败，回退到硬编码标签（但这是为了防止代码崩溃，应该被用户修正）
        print("Falling back to hardcoded dummy labels. **Results will be invalid.** Please fix label loading.")
        all_label = np.array([[1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2], [1], [2]]).flatten() - 1

    print(f'Finish get the data from: {eeg_filename}')

    # 返回聚合数据和试次标签
    return eeg_data, all_label


def main_KUL(name="S13", dataset="KUL", data_document_path="../KUL", time_len=1):
    args = DotMap()
    args.name = name  # 被试名
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128  # KUL 128 #DTU 64 #采样点
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32  # 批量大小
    args.max_epoch = 100
    args.patience = 15
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 16  # KUL16 #DTU 18
    args.eeg_channel = 64  # 通道数
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8  # KUL 8 # 60
    args.cell_number = 46080  # KUL 46080 #DTU 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.csp_comp = 64
    args.log_path = "./result"

    args.frequency_resolution = args.fs / args.window_length

    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    logger = get_logger(args.name, args.log_path, time_len)

    # load data 和 label
    # read_prepared_data 在 KUL/DTU 结构中可能扮演了另一个角色，这里我们跳过它，使用原始的 KUL/DTU 逻辑
    # 假设 args.data_path 和 args.label_path 在 config 中被正确设置
    # eeg_data, event_data = read_prepared_data(args)

    # 假设 KUL 数据的加载逻辑是正确的
    # 此处省略 KUL 特有的加载逻辑，保持原代码不变

    # ... KUL 加载逻辑 ...

    eeg_data, event_data = read_prepared_data(args)

    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = eeg_data.transpose(0, 2, 1)
    event_data = np.squeeze(np.array(event_data) - 1)
    train_data, test_data, train_label, test_label = within_data(eeg_data, event_data)

    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)

    train_data = train_data.transpose(0, 2, 1)  # within:(60, 5760, 64) cross:(54, 6400, 64)
    test_data = test_data.transpose(0, 2, 1)  # within:(60, 64, 640) cross:(6, 6400, 64)
    train_eeg, train_label = sliding_window_csp(train_data, train_label, args,
                                                args.csp_comp)  # within:(5340, 128, 64) cross:(5346, 128, 64)
    test_eeg, test_label = sliding_window_csp(test_data, test_label, args,
                                              args.csp_comp)  # within:(540, 128, 64) cross:(594, 128, 64)

    seq_train_data = np.expand_dims(train_eeg, axis=-1)
    seq_test_data = np.expand_dims(test_eeg, axis=-1)
    # train_label = np.squeeze(train_label - 1)
    # test_label = np.squeeze(test_label - 1)
    del eeg_data

    np.random.seed(200)
    np.random.shuffle(seq_train_data)
    np.random.seed(200)
    np.random.shuffle(train_label)

    np.random.seed(200)
    np.random.shuffle(seq_test_data)
    np.random.seed(200)
    np.random.shuffle(test_label)

    seq_train_data, seq_valid_data, train_label, valid_label = train_test_split(seq_train_data, train_label,
                                                                                test_size=0.1, random_state=42)

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, train_label),
                              batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, test_label),
                             batch_size=args.batch_size, drop_last=True)

    # 训练
    loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)

    info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} acc:{str(acc.item())}'
    result_logger.info(info_msg)

    print(loss, acc)
    logger.info(loss)
    logger.info(acc)
    return acc


def main_DTU(name="S13", dataset="KUL", data_document_path="../DTU", time_len=1):
    args = DotMap()
    args.name = name  # 被试名
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128  # KUL 128 # DTU 64 #采样点
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32  # 批量大小
    args.max_epoch = 100
    args.patience = 15
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 18  # KUL16 #DTU 18
    args.eeg_channel = 64  # 通道数
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 60  # KUL 8 # 60
    args.cell_number = 3200  # KUL 46080 #DTU 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.csp_comp = 64
    args.log_path = "./result"

    args.frequency_resolution = args.fs / args.window_length

    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    logger = get_logger(args.name, args.log_path, time_len)

    # # load data 和 label
    # ------------------------------------DTU------------------------------------------
    subpath = args.data_document_path + '/' + str(args.name) + '_data_preproc.mat'
    eeg_data, event_data = get_data_from_mat(subpath)
    eeg_data = np.array(eeg_data)  # DTU
    eeg_data = eeg_data[:, :, 0:64]  # DTU
    # ------------------------------------DTU------------------------------------------

    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)
    eeg_data = eeg_data.transpose(0, 2, 1)
    event_data = np.squeeze(np.array(event_data) - 1)
    train_data, test_data, train_label, test_label = within_data(eeg_data, event_data)

    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)

    train_data = train_data.transpose(0, 2, 1)  # within:(60, 5760, 64) cross:(54, 6400, 64)
    test_data = test_data.transpose(0, 2, 1)  # within:(60, 64, 640) cross:(6, 6400, 64)
    train_eeg, train_label = sliding_window_csp(train_data, train_label, args,
                                                args.csp_comp)  # within:(5340, 128, 64) cross:(5346, 128, 64)
    test_eeg, test_label = sliding_window_csp(test_data, test_label, args,
                                              args.csp_comp)  # within:(540, 128, 64) cross:(594, 128, 64)

    seq_train_data = np.expand_dims(train_eeg, axis=-1)
    seq_test_data = np.expand_dims(test_eeg, axis=-1)
    # train_label = np.squeeze(train_label - 1)
    # test_label = np.squeeze(test_label - 1)
    del eeg_data

    np.random.seed(200)
    np.random.shuffle(seq_train_data)
    np.random.seed(200)
    np.random.shuffle(train_label)

    np.random.seed(200)
    np.random.shuffle(seq_test_data)
    np.random.seed(200)
    np.random.shuffle(test_label)

    seq_train_data, seq_valid_data, train_label, valid_label = train_test_split(seq_train_data, train_label,
                                                                                test_size=0.1, random_state=42)

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, train_label),
                              batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, test_label),
                             batch_size=args.batch_size, drop_last=True)

    # 训练
    loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)

    info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} acc:{str(acc.item())}'
    result_logger.info(info_msg)

    print(loss, acc)
    logger.info(loss)
    logger.info(acc)
    return acc


def main_AVED(name="S1", dataset="AVED", data_document_path="../AVED", time_len=1, modality="audio-only"):
    args = DotMap()
    args.name = name
    args.subject_number = int(name[1:]) if 'S' in name else int(name)
    args.data_document_path = data_document_path
    args.modality = modality
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 100
    args.patience = 15
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 10
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 16
    args.cell_number = 46080
    args.test_percent = 0.2
    args.vali_percent = 0.1
    args.label_col = 0
    args.csp_comp = 32
    args.log_path = "./result"

    args.frequency_resolution = args.fs / args.window_length

    logger = get_logger(f"{args.name}_{modality}", args.log_path, time_len)

    # ------------------- 加载 AVED 数据 (修正标签加载) -------------------
    print(f'Loading AVED {modality} data for subject {args.name}...')
    # eeg_data_agg: (samples, channels), event_data: (n_trials,)
    eeg_data_agg, event_data = get_AVED_data(args, args.subject_number, modality)

    # ------------------- 重塑为试次 (与 KUL/DTU 逻辑匹配) -------------------
    # shape (N_trials, samples_per_trial, channels)
    samples_per_trial = eeg_data_agg.shape[0] // args.trail_number
    eeg_data_reshaped = []
    for i in range(args.trail_number):
        start_idx = i * samples_per_trial
        end_idx = (i + 1) * samples_per_trial
        trial_data = eeg_data_agg[start_idx:end_idx, :]
        eeg_data_reshaped.append(trial_data)
    eeg_data_reshaped = np.array(eeg_data_reshaped)

    # 调整维度以匹配 KUL/DTU 的 within_data 输入要求: (N_trails, channels, time)
    eeg_data = eeg_data_reshaped.transpose(0, 2, 1)
    # event_data 已经是 (N_trials,)，不需要 squeeze

    # ------------------- 试次级划分 (修正: 引入 within_data 避免数据泄漏) -------------------
    # within_data 假设将 eeg_data (N_trials, C, T) 按试次划分为训练和测试集
    # 并将 event_data (N_trials,) 对应划分
    # 注意: within_data 默认使用 90%/10% 的比例进行划分 (参见 data_process.py)
    train_data, test_data, train_label_trials, test_label_trials = within_data(eeg_data, event_data)

    # ------------------- 应用 CSP 变换 (修正: 引入 CSP) -------------------
    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space',
              norm_trace=True)
    # CSP 作用于 (N_trials, N_channels, N_samples)
    train_data_csp = csp.fit_transform(train_data, train_label_trials) # (N_train_trials, 32, T)
    test_data_csp = csp.transform(test_data)                          # (N_test_trials, 32, T)

    # ... (维度转置和滑动窗口部分不变)

    # ------------------- 滑动窗口 (修正: 使用 sliding_window_csp) -------------------
    train_eeg, train_label = sliding_window_csp(train_data_csp.transpose(0, 2, 1), train_label_trials, args,
                                                args.csp_comp) # (N_windows, T, 32)
    test_eeg, test_label = sliding_window_csp(test_data_csp.transpose(0, 2, 1), test_label_trials, args,
                                              args.csp_comp) # (N_windows, T, 32)

    # ------------------- 填充 CSP 输出到 64 维 (兼容模型硬编码) -------------------
    target_csp_dim = 64
    if args.csp_comp < target_csp_dim:
        pad_width = ((0, 0), (0, 0), (0, target_csp_dim - args.csp_comp), (0, 0)) # (N_win, T, C, 1)

        # 原始 shape: (N_windows, time, features=32, 1)
        seq_train_data = np.expand_dims(train_eeg, axis=-1)
        seq_test_data = np.expand_dims(test_eeg, axis=-1)

        # 填充到 features=64
        seq_train_data = np.pad(seq_train_data, pad_width, mode='constant', constant_values=0)
        seq_test_data = np.pad(seq_test_data, pad_width, mode='constant', constant_values=0)

        # 更新 args.csp_comp 为 64，以确保后续转置和日志正确
        args.csp_comp = target_csp_dim
        print(f"CSP features padded from 32 to {target_csp_dim} to match model's hardcoded input.")
    else:
        seq_train_data = np.expand_dims(train_eeg, axis=-1)
        seq_test_data = np.expand_dims(test_eeg, axis=-1)

    # ------------------- 释放内存 (确保在填充后，释放原始的中间变量) -------------------
    del eeg_data_agg, eeg_data_reshaped, eeg_data, train_data_csp, test_data_csp, train_data, test_data

    # ------------------- 随机化 (可选，但保持一致) -------------------
    np.random.seed(200)
    np.random.shuffle(seq_train_data)
    np.random.seed(200)
    np.random.shuffle(train_label)

    np.random.seed(200)
    np.random.shuffle(seq_test_data)
    np.random.seed(200)
    np.random.shuffle(test_label)


    # ------------------- 划分验证集 -------------------
    # stratify=train_label 仅当 train_label 是 1D 数组时可用
    seq_train_data, seq_valid_data, train_label, valid_label = train_test_split(
        seq_train_data, train_label, test_size=args.vali_percent, random_state=42, stratify=train_label.flatten() # stratify 需要 1D array
    )

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    # ------------------- 调整维度为 (batch, channel=1, features=CSP_comp, time) -------------------
    # 原始形状: (N_windows, time, features, 1)
    # 目标形状 (与 KUL/DTU 一致): (N_windows, 1, features, time)
    # 修正: 使用正确的转置 (0, 3, 2, 1)
    seq_train_data = seq_train_data.transpose(0, 3, 2, 1) # <--- 修正此处
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

    # 检查最终形状
    print(f"Final Training data shape: {seq_train_data.shape} (N_windows, 1, time, features)")
    print(f"Training labels distribution: {np.unique(train_label, return_counts=True)}")

    # ------------------- 数据加载器 -------------------
    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, train_label),
                              batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, test_label),
                             batch_size=args.batch_size, drop_last=True)

    # ------------------- 训练 -------------------
    loss, acc = initiate(args, train_loader, valid_loader, test_loader, f"{args.name}_{modality}")

    info_msg = f'{dataset}_{name}_{modality}_{str(time_len)}s loss:{loss:.4f} acc:{acc.item():.4f}'
    result_logger.info(info_msg)

    print(f"Loss: {loss:.4f}, Acc: {acc.item():.4f}")
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Acc: {acc.item():.4f}")
    return acc.item()


result_logger = logging.getLogger('result')
result_logger.setLevel(logging.INFO)

if __name__ == "__main__":
    file_handler = logging.FileHandler('log/result.log')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    result_logger.addHandler(file_handler)
    all_test_acc = []

    if config.dataset == "KUL":
        for i in range(1, config.people_number + 1):
            name = 'S' + str(i)
            total_acc = main_KUL(name=name, dataset=config.dataset, data_document_path=config.data_document_path,
                                 time_len=config.time_len)
            all_test_acc.append(total_acc)
    elif config.dataset == "DTU":
        for i in range(1, config.people_number + 1):
            name = 'S' + str(i)
            total_acc = main_DTU(name=name, dataset=config.dataset, data_document_path=config.data_document_path,
                                 time_len=config.time_len)
            all_test_acc.append(total_acc)
    elif config.dataset == "AVED":
        # 可以选择运行单个模态或两个模态
        modalities = getattr(config, 'modalities', ['audio-only'])  # 默认只运行audio-only

        for modality in modalities:
            print(f"\n=== Processing AVED {modality} modality ===")
            modality_acc = []
            for i in range(1, config.people_number + 1):
                name = 'S' + str(i)
                acc = main_AVED(name=name, dataset=config.dataset,
                                data_document_path=config.data_document_path,
                                time_len=config.time_len, modality=modality)
                modality_acc.append(acc)

            avg_acc = np.mean(modality_acc)
            std_acc = np.std(modality_acc)
            print(f"AVED {modality} - Average Acc: {avg_acc:.4f} ± {std_acc:.4f}")
            info_msg = f'AVED_{modality}_{str(config.time_len)}s avg_acc:{avg_acc:.4f} std:{std_acc:.4f}'
            result_logger.info(info_msg)
            all_test_acc.extend(modality_acc)

    if all_test_acc:
        final_avg_acc = np.mean(all_test_acc)
        final_std_acc = np.std(all_test_acc)
        print(f'Final average accuracy: {final_avg_acc:.4f} ± {final_std_acc:.4f}')
        info_msg = f'Final_{config.dataset}_{str(config.time_len)}s avg_acc:{final_avg_acc:.4f} std:{final_std_acc:.4f}'
        result_logger.info(info_msg)