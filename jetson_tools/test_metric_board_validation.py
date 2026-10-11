import numpy as np
import pytest
from calibrate_stereo_local import validate_metric_board


def board_view():
    k = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
    c = dict(K1=k, K2=k, D1=np.zeros(5), D2=np.zeros(5), R=np.eye(3),
             T=np.array([[-.1], [0.], [0.]]), expected_baseline_m=.1,
             image_size=np.array([640, 480]), max_vertical_error_px=1.)
    board = np.zeros((54, 3))
    board[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2)*.0254
    xyz = board + [-.1, -.07, 3.]
    left = xyz[:, :2]/xyz[:, 2:]*600 + [320, 240]
    right = left - [20, 0]
    return c, board, left[:, None].astype(np.float32), right[:, None].astype(np.float32)


def test_correct_metric_heldout_board():
    c, board, left, right = board_view()
    assert validate_metric_board(c, [board], [left], [right]) < .001


def test_smooth_but_wrong_disparity_scale_rejected():
    c, board, left, right = board_view()
    right[:, :, 0] += 5
    with pytest.raises(ValueError, match='distances disagree'):
        validate_metric_board(c, [board], [left], [right])


def test_heldout_vertical_misalignment_rejected():
    c, board, left, right = board_view()
    right[:, :, 1] += 2
    with pytest.raises(ValueError, match='vertical mismatch'):
        validate_metric_board(c, [board], [left], [right])
