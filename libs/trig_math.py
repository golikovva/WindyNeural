import numpy as np


def proj_to_trig(x, y):
    speed = np.sqrt(np.square(x) + np.square(y))
    # print(np.argwhere(speed == 0), 'speed == 0')
    sin = np.zeros_like(x)
    cos = np.zeros_like(y)

    eq1 = np.all(np.array([x >= 0, y >= 0]), axis=0)  # x > 0, y > 0, then 3rd quarter
    sin[eq1] = -np.abs(x[eq1]) / speed[eq1]
    cos[eq1] = -np.abs(y[eq1]) / speed[eq1]

    eq2 = np.all(np.array([x >= 0, y < 0]), axis=0)  # x > 0, y < 0, then 4th quarter
    sin[eq2] = -np.abs(x[eq2]) / speed[eq2]
    cos[eq2] = np.abs(y[eq2]) / speed[eq2]

    eq3 = np.all(np.array([x < 0, y >= 0]), axis=0)  # x < 0, y > 0, then 2nd quarter
    sin[eq3] = np.abs(x[eq3]) / speed[eq3]
    cos[eq3] = -np.abs(y[eq3]) / speed[eq3]

    eq4 = np.all(np.array([x < 0, y < 0]), axis=0)  # x < 0, y < 0, then 1st quarter
    sin[eq4] = np.abs(x[eq4]) / speed[eq4]
    cos[eq4] = np.abs(y[eq4]) / speed[eq4]

    # eq0 = np.all(np.array([x == 0, y == 0]))
    sin[np.where(np.isnan(sin))] = 0
    cos[np.where(np.isnan(cos))] = 0
    # print(np.argwhere(np.isnan(sin)), 'found nan indices')

    return speed, sin, cos
