import numpy as np
from scipy.stats import spearmanr

def rank_accuracy(actual_ranks, proxy_ranks):
    """Compare two integer rank arrays and return accuracy metrics.

    Parameters
    ----------
    actual_ranks : array-like of int
        Ground-truth rank for each animal (1 = most dominant).
    proxy_ranks  : array-like of int
        Predicted rank for each animal.

    Returns
    -------
    dict with keys:
        accuracy  – fraction of animals with exact rank match
        within_1  – fraction with |error| <= 1
        within_2  – fraction with |error| <= 2
        mae       – mean absolute error
        rho       – Spearman correlation
    """
    actual = np.asarray(actual_ranks, dtype=float)
    pred   = np.asarray(proxy_ranks,  dtype=float)
    abs_err = np.abs(pred - actual)
    return {
        'accuracy': float(np.mean(abs_err == 0)),
        'within_1': float(np.mean(abs_err <= 1)),
        'within_2': float(np.mean(abs_err <= 2)),
        'mae':      float(np.mean(abs_err)),
        'rho':      float(spearmanr(actual, pred).correlation),
    }




ranks_ground =  ##

ranks1 = [ 1,  4, 12, 11, 10,  8,  2,  9,  6,  5,  7,  3] #cosinor amplitude
ranks2 = [ 2 , 1 ,12,  5, 11,  8,  6 , 9 ,10 , 3 , 7,  4] #Best combo (power_24h + short_gap_frac + night_ibi_cv)
ranks3 = [ 1, 10, 12,  9,  7,  6,  2, 11,  4,  8, 5 , 3] #Best combo (power_24h + high_activity_frac + activity_per_bout)

print(rank_accuracy(ranks1, ranks_ground))
print(rank_accuracy(ranks2, ranks_ground))
print(rank_accuracy(ranks3, ranks_ground))