from types import SimpleNamespace
import numpy as np
import pytest
from calibrate_table_local import neural_pair_depth
from stereo_depth import StereoGeometry


def fixture_pair():
    yy, xx = np.indices((12, 20), dtype=np.float32)
    estimator = SimpleNamespace(image_size=(20, 12), _left_maps=(xx, yy),
                               _right_maps=(xx, yy), left_roi=(0, 0, 20, 12),
                               right_roi=(0, 0, 20, 12),
                               geometry=StereoGeometry(100, 100, 10, 10, .1))
    image = np.zeros((12, 20, 3), np.uint8)
    return estimator, image


def test_metric_scale_and_reverse_rejection():
    estimator, image = fixture_pair()
    left = np.full((12, 20), 5, np.float32)
    right = np.full((12, 20), -5, np.float32)
    matcher = SimpleNamespace(both=lambda a, b: (left, right))
    _, _, depth = neural_pair_depth(estimator, matcher, image, image)
    assert depth[6, 10] == pytest.approx(2)
    assert np.isnan(depth[6, 2])  # correspondence falls outside image
    right[6, 5] = -2
    _, _, depth = neural_pair_depth(estimator, matcher, image, image)
    assert np.isnan(depth[6, 10])  # failed reverse match remains unknown


def test_wrong_resolution_rejected_before_matcher():
    estimator, image = fixture_pair()
    with pytest.raises(ValueError, match='resolution'):
        neural_pair_depth(estimator, None, image[:6], image)
