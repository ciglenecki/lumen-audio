"""To override any variable in config_defaults redefine it in this file.

e.g.
DEFAULT_LR = 0.5
"""


from src.utils.utils_train import MetricMode, OptimizeMetric

DEFAULT_LR = 5e-6
DEFAULT_LR_WARMUP = 1e-4
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 3
DEFAULT_NUM_WORKERS = 4

DEFAULT_OPTIMIZE_METRIC = OptimizeMetric.VAL_F1
DEFAULT_METRIC_MODE = MetricMode.MAX
