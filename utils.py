import numpy as np
import matplotlib.pyplot as plt
import torch


def accuracy(pred, targ):
    pred = torch.softmax(pred, dim=1)
    pred_max_index = torch.max(pred, 1)[1]
    ac = ((pred_max_index == targ).float()).sum().item() / targ.size()[0]
    return ac


def get_link_labels(pos_edge_index, neg_edge_index, device):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def expected_calibration_error(prediction_probabilities, accuracy, confidence):
    """
    Helper function for calculating the expected calibration error as defined in
    the paper On Calibration of Modern Neural Networks, C. Guo, et. al., ICML, 2017

    It is assumed that for a validation dataset, the prediction probabilities have
    been calculated for each point in the dataset and given in the array
    prediction_probabilities.

    Args:
        prediction_probabilities (numpy array):  The predicted probabilities.
        accuracy (numpy array): The accuracy such that the i-th entry in the array holds the proportion of correctly
            classified samples that fall in the i-th bin.
        confidence (numpy array): The confidence such that the i-th entry in the array is the average prediction
            probability over all the samples assigned to this bin.

    Returns:
        float: The expected calibration error.

    """
    if not isinstance(prediction_probabilities, np.ndarray):
        raise ValueError(
            "Parameter prediction_probabilities must be type numpy.ndarray but given object of type {}".format(
                type(prediction_probabilities).__name__
            )
        )
    if not isinstance(accuracy, np.ndarray):
        raise ValueError(
            "Parameter accuracy must be type numpy.ndarray but given object of type {}".format(
                type(accuracy).__name__
            )
        )
    if not isinstance(confidence, np.ndarray):
        raise ValueError(
            "Parameter confidence must be type numpy.ndarray but given object of type {}".format(
                type(confidence).__name__
            )
        )

    if len(accuracy) != len(confidence):
        raise ValueError(
            "Arrays accuracy and confidence should have the same size but instead received {} and {} respectively.".format(
                len(accuracy), len(confidence)
            )
        )

    n_bins = len(accuracy)  # the number of bins
    n = len(prediction_probabilities)  # number of samples
    h = np.histogram(a=prediction_probabilities, range=(0, 1), bins=n_bins)[
        0
    ]  # just the counts
    ece = 0
    for m in np.arange(n_bins):
        ece = ece + (h[m] / n) * np.abs(accuracy[m] - confidence[m])
    return ece


def plot_reliability_diagram(calibration_data, predictions, ece=None, filename=None):
    """
    Helper function for plotting a reliability diagram.

    Args:
        calibration_data (list): The calibration data as a list where each entry in the list is a 2-tuple of type
            numpy.ndarray. Each entry in the tuple holds the fraction of positives and the mean predicted values
            for the true and predicted class labels.
        predictions (np.ndarray): The probabilistic predictions of the classifier for each sample in the dataset used
            for diagnosing miscalibration.
        ece (None or list of float): If not None, this list stores the expected calibration error for each class.
        filename (str or None): If not None, the figure is saved on disk in the given filename.
    """
    if not isinstance(calibration_data, list):
        raise ValueError(
            "Parameter calibration_data should be list of 2-tuples but received type {}".format(
                type(calibration_data).__name__
            )
        )

    if not isinstance(predictions, np.ndarray):
        raise ValueError(
            "Parameter predictions should be of type numpy.ndarray but received type {}".format(
                type(predictions).__name__
            )
        )
    if ece is not None and not isinstance(ece, list):
        raise ValueError(
            "Parameter ece should be None or list of floating point numbers but received type {}".format(
                type(ece).__name__
            )
        )
    if filename is not None and not isinstance(filename, str):
        raise ValueError(
            "Parameter filename should be None or str type but received type {}".format(
                type(filename).__name__
            )
        )

    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=3)
    # ax2 = plt.subplot2grid((6, 1), (4, 0))

    if ece is not None:
        calibration_error = ",".join(format(e, " 0.4f") for e in ece)

    for i, data in enumerate(calibration_data):
        fraction_of_positives, mean_predicted_value = data
        # print(fraction_of_positives, mean_predicted_value)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", alpha=1.0)
        if ece is not None:
            ax1.set_title("Calibration Curve (ECE={})".format(calibration_error))
        ax1.set_xlabel("Confidence", fontsize=16)
        ax1.set_ylabel("Fraction of Positives", fontsize=16)
        ax1.plot([0, 1], [0, 1], "g--")
        # ax2.hist(predictions[:, i], range=(0, 1), bins=10, histtype="step", lw=2)
        # ax2.set_xlabel("Bin", fontsize=16)
        # ax2.set_ylabel("Count", fontsize=16)
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")
