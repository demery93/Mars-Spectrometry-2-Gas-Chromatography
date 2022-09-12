import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from config import config
import time
from contextlib import contextmanager

def aggregated_log_loss(ytrue, ypred):
    scores = []
    for i in range(ytrue.shape[1]):
        scores.append(log_loss(ytrue[:,i], ypred[:,i]))
    return np.mean(scores)

def fill_t_bins(t, intensity, step=config.sample_rate):
    t = np.floor(t / step).astype(int)
    t_min = config.min_time
    t_max = config.max_time

    t_idx = list(range(t_min // step, t_max // step))
    t_bins = np.array([(i + 0.5) * step for i in t_idx])

    bins = []
    for i in t_idx:
        values = intensity[t == i]
        if len(values):
            bins.append(np.mean(values))
        else:
            bins.append(np.nan)

    # bins = [np.mean(a[t == i]) for i in t_idx]

    skip_from = 0
    while skip_from < len(bins) and np.isnan(bins[skip_from]):
        skip_from += 1

    skip_to = len(bins)
    while skip_to > skip_from and np.isnan(bins[skip_to - 1]):
        skip_to -= 1

    t_bins = t_bins[skip_from:skip_to]
    bins = bins[skip_from:skip_to]
    bins = np.array(bins)

    if np.isnan(bins).sum() > 0:
        bins = pd.Series(bins).interpolate().values

    return t_bins, bins

def print_stats(title, array):
    if len(array):
        print(
            "{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}".format(
                title,
                array.shape,
                array.dtype,
                np.min(array),
                np.max(array),
                np.mean(array),
                np.median(array),
            )
        )
    else:
        print(title, "empty")

@contextmanager
def timeit_context(name, enabled=True):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    if enabled:
        print(f"[{name}] finished in {elapsedTime:0.3f}s")