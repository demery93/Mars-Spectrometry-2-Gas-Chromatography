import numpy as np
from sklearn.metrics import log_loss

def aggregated_log_loss(ytrue, ypred):
    scores = []
    for i in range(ytrue.shape[1]):
        scores.append(log_loss(ytrue[:,i], ypred[:,i]))
    return np.mean(scores)