"""Physical metric table fitting and visible sphere-surface depth correction."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from billiards_table import TableCalibration, sample_depth, unproject


WIDTH, HEIGHT, RADIUS, POCKET, INSET = 120.0, 60.0, 2.8575, 11.5, 6.0
SIZE = (1024, 768)
K = np.array([[700.0, 0, 487.0], [0, 715.0, 351.0], [0, 0, 1.0]])
SIGNATURE = {"npz_sha256": "test-only", "baseline_m": 0.127,
             "image_size": list(SIZE), "camera_matrix_hash": "test-camera"}


def rig(oblique=False):
    angle = np.deg2rad(27 if oblique else 0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
    origin = np.array([-0.6, -0.3, 1.65])
    yy, xx = np.indices((SIZE[1], SIZE[0]))
    rays = np.stack(((xx - K[0, 2]) / K[0, 0],
                     (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)), axis=-1)
    normal = rotation[:, 2]
    plane_depth = np.dot(normal, origin) / (rays @ normal)
    local = np.array([[INSET, INSET, 0], [WIDTH - INSET, INSET, 0],
                      [WIDTH - INSET, HEIGHT - INSET, 0], [INSET, HEIGHT - INSET, 0]]) / 100
    camera_markers = local @ rotation.T + origin
    homogeneous = camera_markers @ K.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:]
    return rotation, origin, rays, plane_depth, pixels


def fit(oblique=False, depth_override=None, pixels_override=None, **overrides):
    rotation, origin, rays, depth, pixels = rig(oblique)
    arguments = dict(marker_pixels=pixels if pixels_override is None else pixels_override,
                     depth_map=depth if depth_override is None else depth_override,
                     camera_matrix=K, image_size=SIZE, width_cm=WIDTH, height_cm=HEIGHT,
                     ball_radius_cm=RADIUS, pocket_width_cm=POCKET, inset_cm=INSET,
                     signature=SIGNATURE)
    arguments.update(overrides)
    return TableCalibration.fit(**arguments)


def sphere_scene(oblique=False, x=70, y=33, height=RADIUS):
    rotation, origin, rays, _, _ = rig(oblique)
    center = origin + rotation @ (np.array([x, y, -height]) / 100)
    direction = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    projection = direction @ center
    discriminant = projection ** 2 - (np.dot(center, center) - (RADIUS / 100) ** 2)
    valid = discriminant >= 0
    depth = np.full(rays.shape[:2], np.nan)
    distance = projection[valid] - np.sqrt(discriminant[valid])
    depth[valid] = distance * direction[valid, 2]
    projected = K @ center
    pixel = {"x": float(projected[0] / projected[2]), "y": float(projected[1] / projected[2])}
    pixel_radius = np.sqrt(K[0, 0] * K[1, 1]) * (RADIUS / 100) / center[2]
    return pixel, pixel_radius, depth, center


@pytest.mark.parametrize("oblique", [False, True])
def test_metric_rigid_fit_recovers_pose_and_projected_boundary(oblique):
    calibration = fit(oblique)
    rotation, origin, _, _, pixels = rig(oblique)
    assert calibration.rotation == pytest.approx(rotation, abs=0.001)
    assert calibration.origin == pytest.approx(origin, abs=0.001)
    assert np.linalg.det(calibration.rotation) == pytest.approx(1)
    assert calibration.fit_error_cm < 0.05
    assert calibration.max_dimension_error_cm < 0.1
    assert np.dot(calibration.normal, -calibration.origin) > 0
    assert calibration.table == {"width": WIDTH, "height": HEIGHT,
                                 "ball_radius": RADIUS, "pocket_width": POCKET}
    assert calibration.polygon.shape == (4, 2)
    assert calibration.project(INSET, INSET) == pytest.approx(pixels[0], abs=0.1)


def test_unprojection_uses_actual_vertical_principal_point_and_optical_depth():
    point = unproject({"x": 627, "y": 422.5}, 2.0, K)
    assert point == pytest.approx([0.4, 0.2, 2.0])
    assert unproject([487, 351], 1.2, K) == pytest.approx([0, 0, 1.2])


@pytest.mark.parametrize("oblique", [False, True])
def test_visible_sphere_depth_is_corrected_to_ball_center(oblique):
    calibration = fit(oblique)
    pixel, radius, depth, true_center = sphere_scene(oblique)
    ball = calibration.localize_ball(pixel, radius, depth)
    assert (ball["x"], ball["y"]) == pytest.approx((70, 33), abs=0.15)
    assert ball["height_cm"] == pytest.approx(RADIUS, abs=0.15)
    assert ball["center_depth_m"] == pytest.approx(true_center[2], abs=0.0015)
    assert ball["depth_m"] < ball["center_depth_m"]
    assert ball["depth_source"] == "calibrated_stereo"
    assert ball["localization_source"] == "sphere_center_from_stereo_patch"
    projected = calibration.project(ball["x"], ball["y"], ball["height_cm"])
    assert projected == pytest.approx((pixel["x"], pixel["y"]), abs=1e-8)
    # Without correcting the front surface, the height would be about 2r.
    surface = unproject(pixel, ball["depth_m"], K)
    surface_height = np.dot(surface - calibration.origin, calibration.normal) * 100
    assert surface_height - ball["height_cm"] > 2.0


def test_wrong_stereo_scale_is_rejected_instead_of_fitted_away():
    _, _, _, depth, _ = rig()
    with pytest.raises(ValueError, match="distances disagree"):
        fit(depth_override=depth * 1.1)


def test_one_wrong_marker_depth_fails_metric_calibration():
    _, _, _, depth, pixels = rig()
    point = pixels[2]
    yy, xx = np.indices(depth.shape)
    depth[(xx - point[0]) ** 2 + (yy - point[1]) ** 2 < 36] += 0.08
    with pytest.raises(ValueError, match="disagree|fit error"):
        fit(depth_override=depth)


@pytest.mark.parametrize("order", [[0, 3, 2, 1], [0, 2, 1, 3], [2, 3, 0, 1], [0, 1, 1, 3]])
def test_wrong_corner_order_and_degenerate_clicks_fail(order):
    *_, pixels = rig()
    with pytest.raises(ValueError, match="TL, TR, BR, BL"):
        fit(pixels_override=pixels[order])


def test_calibration_requires_real_valid_depth_and_matching_resolution():
    with pytest.raises(ValueError, match="Insufficient"):
        fit(depth_override=np.full(SIZE[::-1], np.nan))
    with pytest.raises(ValueError, match="resolution"):
        fit(depth_override=np.ones((20, 30)))
    with pytest.raises(ValueError, match="inset"):
        fit(inset_cm=HEIGHT / 2)
    *_, pixels = rig()
    pixels[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit(pixels_override=pixels)


def test_depth_sampler_rejects_sparse_and_mixed_samples_but_tolerates_holes():
    depth = np.ones((31, 31))
    depth[14:17, 14:17] = np.nan
    sampled = sample_depth(depth, (15, 15))
    assert sampled["depth_m"] == 1
    assert 0.7 < sampled["valid_depth_fraction"] < 1
    depth[:, :16] = 1.1
    with pytest.raises(ValueError, match="Mixed"):
        sample_depth(depth, (15, 15))
    depth[:] = np.nan
    depth[15, 15] = 1
    with pytest.raises(ValueError, match="Insufficient"):
        sample_depth(depth, (15, 15))
    with pytest.raises(ValueError, match="outside"):
        sample_depth(depth, (-1, 15))


def test_ball_on_wrong_plane_is_rejected_including_felt_mistaken_for_ball():
    calibration = fit(True)
    pixel, radius, depth, _ = sphere_scene(True, height=RADIUS + 4)
    with pytest.raises(ValueError, match="height"):
        calibration.localize_ball(pixel, radius, depth)
    _, _, _, felt_depth, _ = rig(True)
    pixel, radius, _, _ = sphere_scene(True)
    with pytest.raises(ValueError, match="height"):
        calibration.localize_ball(pixel, radius, felt_depth)


def test_ball_requires_its_own_depth_not_a_plane_intersection_fallback():
    calibration = fit()
    pixel, radius, depth, _ = sphere_scene()
    with pytest.raises(ValueError, match="Insufficient"):
        calibration.localize_ball(pixel, radius, np.full_like(depth, np.nan))
    point_x = int(round(pixel["x"]))
    depth[:, :point_x] += 0.05
    with pytest.raises(ValueError, match="Mixed"):
        calibration.localize_ball(pixel, radius, depth)


def test_out_of_bounds_implausible_size_and_cropped_balls_are_rejected():
    calibration = fit()
    pixel, radius, depth, _ = sphere_scene(x=1)
    with pytest.raises(ValueError, match="bounds"):
        calibration.localize_ball(pixel, radius, depth)
    pixel, radius, depth, _ = sphere_scene()
    with pytest.raises(ValueError, match="physical size"):
        calibration.localize_ball(pixel, radius * 3, depth)
    with pytest.raises(ValueError, match="too small"):
        calibration.localize_ball(pixel, 1, depth)
    with pytest.raises(ValueError, match="partially outside"):
        calibration.localize_ball({"x": 1, "y": 100}, radius, depth)


def test_atomic_json_round_trip_and_rig_signature_binding(tmp_path):
    calibration = fit(True)
    path = tmp_path / "profiles" / "table.json"
    calibration.save(path)
    restored = TableCalibration.load(path, SIGNATURE)
    assert restored.to_dict() == calibration.to_dict()
    assert restored.polygon == pytest.approx(calibration.polygon)
    assert list(path.parent.iterdir()) == [path]
    # Existing profile can be atomically replaced too.
    calibration.save(path)
    for key, value in (("baseline_m", 0.2), ("npz_sha256", "moved-rig"),
                       ("image_size", [640, 480]), ("camera_matrix_hash", "other")):
        with pytest.raises(ValueError, match="signature"):
            TableCalibration.load(path, {**SIGNATURE, key: value})


@pytest.mark.parametrize("corruption", ["rotation", "normal", "matrix", "version"])
def test_invalid_stored_geometry_cannot_bypass_validation(corruption):
    state = fit().to_dict()
    if corruption == "rotation":
        state["rotation"][0][0] = 2
    elif corruption == "normal":
        state["normal_sign"] *= -1
    elif corruption == "matrix":
        state["camera_matrix"][1][1] = 0
    else:
        state["version"] = 999
    with pytest.raises(ValueError):
        TableCalibration.from_dict(state)
