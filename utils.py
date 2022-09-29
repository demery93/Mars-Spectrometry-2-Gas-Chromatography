import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from config import config
import time
from contextlib import contextmanager
import torch
import math
import warnings
import tensorflow as tf
import yaml

def load_config_data(experiment_name: str, experiments_dir='experiments') -> dict:
    with open(f"{experiments_dir}/{experiment_name}.yaml") as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def aggregated_log_loss(ytrue, ypred):
    scores = []
    for i in range(ytrue.shape[1]):
        scores.append(log_loss(ytrue[:,i], ypred[:,i]))
    return np.mean(scores)

def fill_t_bins(t, intensity, step=config.sample_rate):
    t = np.floor(t / step).astype(int)
    t_min = config.min_time
    t_max = config.max_time

    t_idx = list(range(int(t_min // step), int(t_max // step)))
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


class CosineAnnealingWarmRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule with restarts.
    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.
    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies a cosine decay function with
    restarts to an optimizer step, given a provided initial learning rate.
    It requires a `step` value to compute the decayed learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    The learning rate multiplier first decays
    from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
    restart is performed. Each new warm restart runs for `t_mul` times more
    steps and with `m_mul` times initial learning rate as the new learning rate.
    Example usage:
    ```python
    first_decay_steps = 1000
    lr_decayed_fn = (
      tf.keras.optimizers.schedules.CosineDecayRestarts(
          initial_learning_rate,
          first_decay_steps))
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        first_decay_steps,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.0,
        name=None,
    ):
        """Applies cosine decay with restarts to the learning rate.
        Args:
          initial_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
            number. Number of steps to decay over.
          t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the number of iterations in the i-th period.
          m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
            Used to derive the initial learning rate of the i-th period.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of the
            initial_learning_rate.
          name: String. Optional name of the operation. Defaults to 'SGDRDecay'.
        """
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self._t_mul = t_mul
        self._m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "SGDRDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)

            global_step_recomp = tf.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = tf.floor(
                        tf.math.log(1.0 - completed_fraction * (1.0 - t_mul))
                        / tf.math.log(t_mul)
                    )

                    sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                    completed_fraction = (
                        completed_fraction - sum_r
                    ) / t_mul**i_restart

                else:
                    i_restart = tf.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction

            i_restart, completed_fraction = tf.cond(
                tf.equal(t_mul, 1.0),
                lambda: compute_step(completed_fraction, geometric=False),
                lambda: compute_step(completed_fraction, geometric=True),
            )

            m_fac = m_mul**i_restart
            cosine_decayed = (
                0.5
                * m_fac
                * (
                    1.0
                    + tf.cos(
                        tf.constant(math.pi, dtype=dtype) * completed_fraction
                    )
                )
            )
            decayed = (1 - alpha) * cosine_decayed + alpha

            return tf.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": self.name,
        }


def check_CosineAnnealingWarmRestarts():
    import matplotlib.pyplot as plt

    optimizer = tf.optimizers.Adam(learning_rate=1)
    scheduler = CosineAnnealingWarmRestarts(initial_learning_rate=1, first_decay_steps=16, t_mul=1.41421)

    lrs = []
    for _ in range(110):
        lrs.append(scheduler.lr)

    # 251: 77
    # 371: 49
    # 536: 37
    # 771: 27
    # 1101: 17
    # 1536: 13

    plt.plot(lrs, label='Relative learning rate')
    plt.scatter([16, 38, 68, 110], [0]*4, c='r', label='False positive mining')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    check_CosineAnnealingWarmRestarts()