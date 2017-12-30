import numpy as np

eps = np.finfo(np.float32).eps


def reject_outliers_noop(data):
    selected = np.where(np.ones_like(data) == 1)
    return selected


def reject_outliers2(data, m=2, min_val=None):
    mean = np.mean(data)

    if min_val is not None:
        mean = np.mean(data[data > min_val])

    selector = np.where(abs(data - mean) <= m * np.std(data))
    return selector


def reject_outliers(data, m=1.5, min_val=None):
    median = np.median(data)
    if min_val is not None:
        median = np.median(data[data > min_val])

    d = np.abs(data - median)
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    selector = np.where(s <= m)

    return selector


def get_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1 + eps)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def line2pts(lines):
    pts = []
    for x1, y1, x2, y2 in lines:
        pts.append((x1, y1))
        pts.append((x2, y2))

    return pts


def extrapolate_line(line_param, top_y, bottom_y, state=None):
    assert not (state is None and line_param is None)

    m = None
    b = None

    if line_param is not None:
        vx, vy, x, y = line_param
        m = vy / vx
        b = int(y - x * m)

    if state is not None:
        state['m'] = state.get('m', [])
        state['b'] = state.get('b', [])
        _ms = state['m']
        _bs = state['b']

        if m is None:
            m = _ms[-1]
            b = _bs[-1]
        else:
            _ms.append(m)
            _bs.append(b)
            _ms = _ms[:-20]
            _bs = _bs[:-20]

        if len(_ms) > 7:
            m = running_mean(_ms, 5)[-2]
            b = running_mean(_bs, 5)[-2]

    # print('y = %sx + %s' % (m, b))
    # print('==')
    top_x = int((top_y - b) / m)
    bottom_x = int((bottom_y - b) / m)

    return (top_x, int(top_y), bottom_x, int(bottom_y)), state


def binary_img_2_rgb(img):
    img = np.tile(img[..., None], 3)
    return img
