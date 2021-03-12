import numpy as np


def standard_error(data: np.ndarray) -> float:
    return np.std(data, ddof=1) / np.sqrt(np.size(data))
