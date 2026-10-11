"""Four physical spheres calibrate cushion coordinates and the felt plane."""

import copy

import numpy as np
import pytest

from billiards_table import TableCalibration


WIDTH, HEIGHT, RADIUS, POCKET = 120.0, 60.0, 2.8575, 11.5
GAP = 7.62
SIZE = (1024, 768)
K = np.array([[700.0, 0.0, 487.0], [0.0, 715.0, 351.0], [0.0, 0.0, 1.0]])
SIGNATURE = {"camera": "calibrated-left", "size": list(SIZE)}


def scene(angle=27, kind="camera_z", reference="surface"):
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    origin = np.array([-0.6, -0.3, 1.65])
    inset = GAP + RADIUS
    coordinates = [[inset, inset], [WIDTH - inset, inset],
                   [WIDTH - inset, HEIGHT - inset], [inset, HEIGHT - inset]]
    balls, centers = [], []
    for name, (x, y) in zip(("TL", "TR", "BR", "BL"), coordinates):
        center = origin + rotation @ (np.array([x, y, -RADIUS]) / 100)
        centers.append(center)
        balls.append(observation(center, kind, reference, name=name))
    return balls, rotation, origin, centers


def observation(center, kind, reference, **extra):
    measured = center.copy()
    if reference == "surface":
        measured -= RADIUS / 100 * center / np.linalg.norm(center)
    projected = K @ center
    return {"pixel": (projected[:2] / projected[2]).tolist(),
            "distance_m": float(measured[2] if kind == "camera_z" else np.linalg.norm(measured)),
            "distance_kind": kind, "measurement_reference": reference,
            "radius_px": float(np.sqrt(K[0, 0] * K[1, 1]) * RADIUS / 100 / center[2]),
            "camera_id": "left", "frame_id": 123, "calibration_signature": SIGNATURE,
            **extra}


def fit(balls, **overrides):
    arguments = dict(balls=balls, camera_matrix=K, image_size=SIZE, width_cm=WIDTH,
                     height_cm=HEIGHT, ball_radius_cm=RADIUS, pocket_width_cm=POCKET,
                     signature=SIGNATURE, max_error_cm=0.25)
    arguments.update(overrides)
    return TableCalibration.fit_ball_measurements(**arguments)


@pytest.mark.parametrize("angle", [0, 27, 65])
@pytest.mark.parametrize("kind", ["camera_z", "range"])
@pytest.mark.parametrize("reference", ["surface", "center"])
def test_four_spheres_recover_felt_pose_and_three_inch_surface_gap(angle, kind, reference, tmp_path):
    balls, rotation, origin, _ = scene(angle, kind, reference)
    calibration = fit(balls)
    assert calibration.origin == pytest.approx(origin, abs=1e-12)
    assert calibration.rotation == pytest.approx(rotation, abs=1e-12)
    assert calibration.inset_cm == pytest.approx(GAP + RADIUS)
    assert calibration.reference_type == "balls"
    assert calibration.fit_error_cm < 1e-10
    assert calibration.max_dimension_error_cm < 1e-10
    # The origin must be on the felt, not erroneously on the center plane.
    projected_origin = K @ origin
    assert calibration.project(0, 0) == pytest.approx(projected_origin[:2] / projected_origin[2])
    assert calibration.project(GAP + RADIUS, GAP + RADIUS, RADIUS) == pytest.approx(balls[0]["pixel"])
    path = tmp_path / "table.json"
    calibration.save(path)
    restored = TableCalibration.load(path, SIGNATURE)
    assert restored.to_dict() == calibration.to_dict()
    center = origin + rotation @ (np.array([84.0, 35.0, -RADIUS]) / 100)
    measured = observation(center, kind, reference)
    localized = restored.localize_ball_measurement(
        measured["pixel"], measured["distance_m"], distance_kind=kind,
        measurement_reference=reference, radius_px=measured["radius_px"])
    assert [localized["x"], localized["y"], localized["height_cm"]] == pytest.approx([84, 35, RADIUS])


def sphere_depth(centers):
    yy, xx = np.indices((SIZE[1], SIZE[0]))
    rays = np.stack(((xx - K[0, 2]) / K[0, 0],
                     (yy - K[1, 2]) / K[1, 1], np.ones_like(xx)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    depth = np.full((SIZE[1], SIZE[0]), np.nan)
    for center in centers:
        projection = rays @ center
        discriminant = projection ** 2 - (center @ center - (RADIUS / 100) ** 2)
        visible = discriminant >= 0
        depth[visible] = (projection[visible] - np.sqrt(discriminant[visible])) * rays[visible, 2]
    return depth


@pytest.mark.parametrize("with_radii", [False, True])
def test_stereo_ball_surfaces_fit_the_same_felt_plane(with_radii):
    balls, rotation, origin, centers = scene(angle=55)
    calibration = TableCalibration.fit_balls(
        [ball["pixel"] for ball in balls], sphere_depth(centers), K, SIZE,
        WIDTH, HEIGHT, RADIUS, POCKET, SIGNATURE,
        ball_radii_px=[ball["radius_px"] for ball in balls] if with_radii else None,
        max_error_cm=0.25)
    assert calibration.origin == pytest.approx(origin, abs=0.002)
    assert calibration.rotation == pytest.approx(rotation, abs=0.002)
    assert calibration.reference_type == "balls"
    center = origin + rotation @ (np.array([80.0, 40.0, -RADIUS]) / 100)
    measured = observation(center, "range", "center")
    localized = calibration.localize_ball_measurement(
        measured["pixel"], measured["distance_m"], distance_kind="range", measurement_reference="center")
    assert [localized["x"], localized["y"], localized["height_cm"]] == pytest.approx([80, 40, RADIUS], abs=0.2)


@pytest.mark.parametrize("count", [0, 3, 5])
def test_rejects_other_than_four_reference_balls(count):
    balls, _, _, _ = scene()
    balls = (balls + [balls[0]])[:count]
    with pytest.raises(ValueError, match="exactly four"):
        fit(balls)


@pytest.mark.parametrize("field", ["pixel", "distance_m", "distance_kind", "measurement_reference"])
def test_each_ball_requires_complete_measurement_semantics(field):
    balls, _, _, _ = scene()
    balls[0].pop(field)
    with pytest.raises(ValueError, match="Each calibration ball"):
        fit(balls)


@pytest.mark.parametrize("field,value", [("frame_id", 124), ("camera_id", "right"),
                                          ("calibration_signature", {"camera": "other"})])
def test_rejects_mixed_frames_cameras_and_calibrations(field, value):
    balls, _, _, _ = scene()
    balls[2][field] = value
    with pytest.raises(ValueError, match=field):
        fit(balls)


def test_rejects_wrong_corner_order_and_inconsistent_distances():
    balls, _, _, _ = scene()
    with pytest.raises(ValueError, match="Ball names"):
        fit([balls[0], balls[2], balls[1], balls[3]])
    balls[2]["distance_m"] += 0.10
    with pytest.raises(ValueError, match="distances disagree"):
        fit(balls)


def test_no_depth_or_table_plane_fallback_for_missing_ball_depth():
    balls, _, _, _ = scene()
    with pytest.raises(ValueError, match="Insufficient valid stereo depth"):
        TableCalibration.fit_balls([ball["pixel"] for ball in balls],
                                  np.full((SIZE[1], SIZE[0]), np.nan), K, SIZE,
                                  WIDTH, HEIGHT, RADIUS, POCKET, SIGNATURE)


@pytest.mark.parametrize("radius_px", [None, 0, 2, float("nan"), 60])
def test_invalid_ball_radius_measurements_reject_alignment(radius_px):
    balls, _, _, _ = scene()
    balls[0]["radius_px"] = radius_px
    with pytest.raises(ValueError, match="radius"):
        fit(balls)


def test_legacy_marker_profile_remains_readable_without_reference_type():
    profile = fit(scene()[0]).to_dict()
    profile.pop("reference_type")
    assert TableCalibration.from_dict(profile).reference_type == "markers"
    malformed = copy.deepcopy(profile)
    malformed["reference_type"] = "unknown"
    with pytest.raises(ValueError, match="reference type"):
        TableCalibration.from_dict(malformed)
