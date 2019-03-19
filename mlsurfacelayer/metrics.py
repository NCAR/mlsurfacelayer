import numpy as np


def pearson_r2(y_true, y_pred):
    """
    Calculate the pearson correlation coefficient squared.

    Args:
        y_true:
        y_pred:

    Returns:

    """
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2


def mean_error(y_true, y_pred):
    """
    Calculate the mean error.

    Args:
        y_true:
        y_pred:

    Returns:

    """
    return np.mean(y_pred - y_true)


def hellinger_distance(y_true, y_pred, y_bin_count=30):
    """
    Calculate hellinger distance, an integrated measure of the differences between 2 pdfs.

    Args:
        y_true: array of true values
        y_pred: array of predicted values
        y_bin_count: number of bins for discretizing the pdfs. Assumes linear spacing
            between bins.
    Returns:
        float: hellinger distance
    """
    y_bins = np.linspace(np.minimum(y_true.min(), y_pred.min()),
                         np.maximum(y_true.max(), y_pred.max()),
                         y_bin_count)
    y_bin_centers = 0.5 * (y_bins[1:] + y_bins[:-1])
    y_true_hist, _ = np.histogram(y_true, bins=y_bins, density=True)
    y_pred_hist, _ = np.histogram(y_pred, bins=y_bins, density=True)
    pdf_distances = (np.sqrt(y_true_hist) - np.sqrt(y_pred_hist)) ** 2
    return np.trapz(pdf_distances, y_bin_centers) / 2
