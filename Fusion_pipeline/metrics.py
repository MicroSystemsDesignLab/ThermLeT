from sklearn.metrics import mean_squared_error
import torch
import numpy as np

def eval_metrics(output, target):
    metrics = [mse, mean_error, abs_rel_diff, scale_invariant_error, median_error,  rms_linear]
    acc_metrics = np.zeros(len(metrics))
    output = output.cpu().data.numpy()
    target = target.cpu().data.numpy()
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics

def rms_linear(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    is_nan = np.isnan(abs_diff)
    return np.sqrt((abs_diff[~is_nan]**2).mean())

def mean_error(y_input, y_target):
    abs_diff = np.abs(y_target-y_input)
    return abs_diff[~np.isnan(abs_diff)].mean()

def mse(y_input, y_target):
    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    sum_mse_over_batch = 0.

    for i in range(N):
        sum_mse_over_batch += mean_squared_error(
            y_input[i, 0, :, :][~np.isnan(y_target[i, 0, :, :])], y_target[i, 0, :, :][~np.isnan(y_target[i, 0, :, :])])

        if C == 3:  # color
            sum_mse_over_batch += mean_squared_error(
                y_input[i, 1, :, :][~np.isnan(y_target[i, 1, :, :])], y_target[i, 1, :, :][~np.isnan(y_target[i, 1, :, :])])
            sum_mse_over_batch += mean_squared_error(
                y_input[i, 2, :, :][~np.isnan(y_target[i, 2, :, :])], y_target[i, 2, :, :][~np.isnan(y_target[i, 2, :, :])])

    mean_mse = sum_mse_over_batch / (float(N))
    if C == 3:
        mean_mse /= 3.0

    return mean_mse
