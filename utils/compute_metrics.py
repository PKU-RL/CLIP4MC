import numpy as np


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    return {'R1': float(np.sum(ind == 0)) * 100 / len(ind),
            'R5': float(np.sum(ind < 5)) * 100 / len(ind),
            'R10': float(np.sum(ind < 10)) * 100 / len(ind),
            'MedianR': np.median(ind) + 1,
            'MeanR': np.mean(ind) + 1,
            }
    # metrics["cols"] = [int(i) for i in list(ind)]
