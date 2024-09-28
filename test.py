import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# RUN TRAINING

def mse(
    y: np.ndarray,
    y_hat: np.ndarray,
    weights=None,
    axis=None,
):
    # _metric_protections(y, y_hat, weights)

    delta_y = np.square(y - y_hat)
    if weights is not None:
        mse = np.average(
            delta_y[~np.isnan(delta_y)], weights=weights[~np.isnan(delta_y)], axis=axis
        )
    else:
        mse = np.nanmean(delta_y, axis=axis)

    return mse


def mqloss(
    y: np.ndarray,
    y_hat: np.ndarray,
    quantiles: np.ndarray,
    weights=None,
    axis=None,
):
    if weights is None:
        weights = np.ones(y.shape)

    n_q = len(quantiles)

    y_rep = np.expand_dims(y, axis=-1)
    error = y_hat - y_rep
    sq = np.maximum(-error, np.zeros_like(error))
    s1_q = np.maximum(error, np.zeros_like(error))
    mqloss = quantiles * sq + (1 - quantiles) * s1_q

    # Match y/weights dimensions and compute weighted average
    weights = np.repeat(np.expand_dims(weights, axis=-1), repeats=n_q, axis=-1)
    mqloss = np.average(mqloss, weights=weights, axis=axis)

    return mqloss


# seq_len = 288
# pred_len = [288, 432, 576]
seq_len = 96
pred_len = [96, 192, 336, 720]
root_dir = "/home/user/data/THU-timeseries"
data_dir = [
    ("electricity", "electricity/electricity.csv"),
    ("etth1", "ETT-small/ETTh1.csv"),
    ("etth2", "ETT-small/ETTh2.csv"),
    ("ettm1", "ETT-small/ETTm1.csv"),
    ("ettm2", "ETT-small/ETTm2.csv"),
    ("exchange_rate", "exchange_rate/exchange_rate.csv"),
    ("traffic", "traffic/traffic.csv"),
    # ("mfred", "MFRED/MFRED.csv"),
    ("weather", "weather/weather.csv"),
]

# for pl in pred_len:
#     for dt, d in data_dir:
#         d_dir = os.path.join(root_dir, d)
#         os.system(
#             f"python exe_forecasting.py --datatype {dt} --device cuda:1 --data_dir {d_dir} --pred_len {pl} --seq_len {seq_len} --seed 4"
#         )

# seq_len = 288
# pred_len = [288, 432, 576]
# # seq_len = 96
# # pred_len = [96, 192, 336, 720]
# root_dir = "/home/user/data/THU-timeseries"
# data_dir = [
#     # ("electricity", "electricity/electricity.csv"),
#     # ("etth1", "ETT-small/ETTh1.csv"),
#     # ("etth2", "ETT-small/ETTh2.csv"),
#     # ("ettm1", "ETT-small/ETTm1.csv"),
#     # ("ettm2", "ETT-small/ETTm2.csv"),
#     # ("exchange_rate", "exchange_rate/exchange_rate.csv"),
#     # ("traffic", "traffic/traffic.csv"),
#     ("mfred", "MFRED/MFRED.csv"),
#     # ("weather", "weather/weather.csv"),
# ]

# for i in range(5):
#     for pl in pred_len:
#         for dt, d in data_dir:
#             d_dir = os.path.join(root_dir, d)
#             os.system(
#                 f"python exe_forecasting.py --datatype {dt} --device cuda:1 --data_dir {d_dir} --pred_len {pl} --seq_len {seq_len} --seed {i}"
#             )


# COLLECT METRICS
all_df = []
for dt, d in data_dir:
    df = []
    for pl in pred_len:
        print(dt, pl)
        # TODO: 5 runs
        seed_results = []
        for ii in range(5):
            try:
                result_pth = os.path.join(f'save/forecasting_{dt}_{ii}_{pl}', 'result_nsample100.pk')
                with open(result_pth, 'rb') as f:
                    results = np.array(pickle.load(f))
            except:
                result_pth = os.path.join(f'save/forecasting_{dt}_{ii}_{pl}', 'generated_outputs_nsample100.pk')
                with open(result_pth, 'rb') as f:
                    results = pickle.load(f)
                    
                all_generated_samples, all_target = results
                quantiles = (np.arange(9) + 1) / 10
                y_pred = all_generated_samples.transpose((1, 0, 2, 3))
                y_pred_point = np.mean(y_pred, axis=0)[:, -pl:, :]
                y_pred_q = np.quantile(y_pred, quantiles, axis=0)
                y_pred_q = np.transpose(y_pred_q, (1, 2, 3, 0))[:, -pl:, :, :]
                y_real = all_target[:, -pl:, :]


                MSE = mse(y_real, y_pred_point)
                CRPS = mqloss(y_real, y_pred_q, quantiles=np.array(quantiles))
                results = np.array([MSE, CRPS])
                
            seed_results.append(results.reshape(1,-1))
        seed_results = np.concatenate(seed_results)
        print(seed_results)
        df.append(seed_results.mean(axis=0, keepdims=True))
    df = np.concatenate(df)
    df = pd.DataFrame(df, columns=["MSE", "CRPS"])
    df['dataset'] = dt
    df['pred_len'] = pred_len
    all_df.append(df)
all_df = pd.concat(all_df)
all_df = all_df[["dataset", "pred_len", "MSE", 'CRPS']]
all_df.to_csv('csdi_forecast_result.csv')
