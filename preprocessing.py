import os
from get_args import Args
import numpy as np

# https://github.com/microsoft/anomalydetector/blob/master/msanomalydetector/spectral_residual.py
def average_filter(values, n):
    """
    Calculate the sliding window average for the give time series.
    Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
    :param values: list.
        a list of float numbers
    :param n: int, default 3.
        window size.
    :return res: list.
        a list of value after the average_filter process.
    """

    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float, axis=0) # axis plus

    res[n:] = res[n:] - res[:-n]

    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res

def spectral_residual_transform(values, args):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """
    EPS = 1e-8
    __mag_window = args.seq_len
    ##
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(mag <= EPS)[0]
    mag[eps_index] = EPS
    
    mag_log = np.log(mag)
    mag_log[eps_index] = 0
    spectral = np.exp(mag_log - average_filter(mag_log, n=__mag_window))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
    return mag

if __name__ == "__main__":
    args_class = Args()
    args = args_class.args
    data_folder = args.data_path + args.dataset
    data_list = os.listdir(data_folder)
    if args.choice_data:
        data_list = [i for i in data_list if any(key in i for key in args.choice_data)]
    train_list = [i for i in data_list if 'train' in i]; train_list.sort()
    train_data = []
    
    cnt = 0
    for f in range(len(train_list)):
        load_file = np.load(os.path.join(data_folder, train_list[f]))
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(15,6))
        # plt.plot(load_file[:,0])
        # plt.savefig(f"Fig/train_{f}.jpg")

        load_file = spectral_residual_transform(load_file, args)
        # plt.clf()
        # plt.figure(figsize=(15,6))
        # plt.plot(load_file[:,0])
        # plt.savefig(f"Fig/train2_{f}.jpg")
        # print(np.where(load_file!=0))
        train_data.append(load_file)

    # data_x_2d = np.concatenate(train_data)
    # print(train_data[0].shape)