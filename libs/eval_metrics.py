import numpy as np


def weight_speed_vector(weighting_speed, true_speed):
    assert weighting_speed.shape == true_speed.shape, 'be sure vectors have the same shape'
    weighting_speed[true_speed[:, :] < 4] = weighting_speed[true_speed[:, :] < 4] * 0.25
    weighting_speed[np.multiply(true_speed[:, :] >= 4, true_speed[:, :] < 6)] = weighting_speed[
                                                                                    np.multiply(true_speed[:, :] >= 4,
                                                                                                true_speed[:,
                                                                                                :] < 6)] * 0.7
    weighting_speed[np.multiply(true_speed[:, :] >= 6, true_speed[:, :] < 8)] = weighting_speed[
                                                                                    np.multiply(true_speed[:, :] >= 6,
                                                                                                true_speed[:,
                                                                                                :] < 8)] * 1
    weighting_speed[np.multiply(true_speed[:, :] >= 8, true_speed[:, :] < 10)] = weighting_speed[
                                                                                     np.multiply(true_speed[:, :] >= 8,
                                                                                                 true_speed[:,
                                                                                                 :] < 10)] * 0.8
    weighting_speed[np.multiply(true_speed[:, :] >= 10, true_speed[:, :] < 12)] = weighting_speed[np.multiply(
        true_speed[:, :] >= 10, true_speed[:, :] < 12)] * 0.6
    weighting_speed[true_speed[:, :] >= 12] = weighting_speed[true_speed[:, :] >= 12] * 0.3
    return weighting_speed


def calc_wRMSE(y_pred, y_true):
    weight_speed_vector(y_pred, y_true)
    weight_speed_vector(y_true, y_true)
    wRMSE = np.sqrt((np.square(y_pred - y_true)).mean(axis=0))
    print("RMSE_weighted +1/+6h:", wRMSE)


def speed_to_category(speed):
    speed[speed[:, :] < 4] = 1
    speed[np.multiply(speed[:, :] >= 4, speed[:, :] < 6)] = 2
    speed[np.multiply(speed[:, :] >= 6, speed[:, :] < 8)] = 3
    speed[np.multiply(speed[:, :] >= 8, speed[:, :] < 10)] = 4
    speed[np.multiply(speed[:, :] >= 10, speed[:, :] < 12)] = 5
    speed[speed[:, :] >= 12] = 6


def category_to_delta_category(speed_category):
    for i in range(speed_category.shape[0] - 1, 0, -1):
        for j in range(speed_category.shape[1]):
            speed_category[i][j] = speed_category[i][j] - speed_category[i - 1][j]


def category_metric(true_metric_delta, pred_metric_delta):
    return (np.fabs(true_metric_delta - pred_metric_delta)).mean(axis=0)


def calc_cat_change_metric(y_pred, y_true):
    speed_to_category(y_pred)
    speed_to_category(y_true)
    category_to_delta_category(y_pred)
    category_to_delta_category(y_true)
    print("category_change_metric +1/+6h:", category_metric(y_true, y_pred))
