"""Metric table coordinates from calibrated camera pixels and metric distance.

Four flat markers or four resting balls reference the inner cushion corners.
Their measured camera positions fit a rigid transform, never an adjustable
scale or a depth-free homography. That transform maps stereo depth or explicit
metric camera observations into table coordinates exposed to the planner in cm.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _pixel(pixel):
    try:
        value = [pixel["x"], pixel["y"]] if isinstance(pixel, dict) else pixel
        value = np.asarray(value, dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Pixel must contain finite x and y coordinates") from exc
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("Pixel must contain finite x and y coordinates")
    return value


def _camera_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-9)
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError("Camera matrix must be a finite, invertible pinhole matrix")
    return matrix.copy()


def _image_size(value):
    size = np.asarray(value, dtype=float)
    if (size.shape != (2,) or not np.isfinite(size).all()
            or np.any(size < 1) or np.any(size != np.rint(size))):
        raise ValueError("Image size must contain positive integer width and height")
    return tuple(int(v) for v in size)


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (ValueError, TypeError) as exc:
        raise ValueError("Calibration signature must be finite JSON-compatible data") from exc


def sample_depth(depth_map, pixel, radius_px=4.0, min_valid=5,
                 min_valid_fraction=0.4, max_iqr_m=0.01):
    """Return a valid central disk's median *raw stereo Z* and quality metrics.

    Missing, sparse, or mixed depth raises ValueError; there is no substitute
    derived from a table plane or the expected physical size of an object.
    """
    depth = np.asarray(depth_map, dtype=float)
    point = _pixel(pixel)
    if depth.ndim != 2 or min(depth.shape, default=0) < 1:
        raise ValueError("Stereo depth must be a nonempty 2D array")
    if (not np.isfinite(radius_px) or radius_px <= 0
            or isinstance(min_valid, bool) or not isinstance(min_valid, (int, np.integer))
            or min_valid < 1
            or not 0 < min_valid_fraction <= 1
            or not np.isfinite(max_iqr_m) or max_iqr_m < 0):
        raise ValueError("Invalid stereo depth sampling thresholds")
    height, width = depth.shape
    x, y = point
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Pixel is outside the stereo depth image")
    xa, xb = max(0, int(np.floor(x - radius_px))), min(width, int(np.ceil(x + radius_px)) + 1)
    ya, yb = max(0, int(np.floor(y - radius_px))), min(height, int(np.ceil(y + radius_px)) + 1)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius_px ** 2
    patch = depth[ya:yb, xa:xb][disk]
    values = patch[np.isfinite(patch) & (patch > 0)]
    fraction = float(values.size / patch.size) if patch.size else 0.0
    if values.size < min_valid or fraction < min_valid_fraction:
        raise ValueError("Insufficient valid stereo depth in the sampling disk")
    spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    if spread > max_iqr_m:
        raise ValueError("Mixed or inconsistent stereo depth in the sampling disk")
    return {"depth_m": float(np.median(values)), "valid_depth_fraction": fraction,
            "depth_iqr_m": spread, "valid_depth_count": int(values.size)}


def unproject(pixel, depth_m, camera_matrix):
    """Convert a rectified pixel and metric camera-axis Z to camera XYZ."""
    point = _pixel(pixel)
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("Unprojection requires finite positive stereo depth")
    ray = np.linalg.solve(_camera_matrix(camera_matrix), [point[0], point[1], 1.0])
    return ray * (float(depth_m) / ray[2])


def _measurement_number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return float(value)


def unproject_measurement(pixel, distance_m, camera_matrix, *, distance_kind):
    """Unproject a rectified pixel and an explicitly typed metric distance.

    ``camera_z`` is distance along the optical axis; ``range`` is Euclidean
    distance from the camera center along the pixel's ray. They differ off
    axis and must never be inferred from the numeric value. Inputs must already
    be calibrated meters, not normalized or relative model predictions.
    """
    distance = _measurement_number(distance_m, "Metric distance in meters", positive=True)
    if distance_kind not in ("camera_z", "range"):
        raise ValueError("Distance kind must explicitly be 'camera_z' or 'range'")
    ray = unproject(pixel, 1.0, camera_matrix)
    camera = ray * (distance if distance_kind == "camera_z" else distance / np.linalg.norm(ray))
    if not np.isfinite(camera).all() or camera[2] <= 0:
        raise ValueError("Metric measurement must produce a finite camera point in front of the camera")
    return camera


def camera_ball_separation(first, second, camera_matrix, *, ball_radius_cm):
    """Measure center distance/gap from two observations in one camera frame.

    No camera-to-table pose, pocket coordinates, or camera height is required.
    Each observation must supply a rectified pixel, metric distance, explicit
    distance_kind ('camera_z'/'range'), and measurement_reference ('surface'/
    'center'). Camera intrinsics must match those pixels. The result is a 3D
    distance; for equal balls resting on a level table it is also their planar
    center separation. This alone cannot establish pocket locations or table axes.
    """
    radius_m = _measurement_number(ball_radius_cm, "Ball radius in centimeters", positive=True) / 100
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("Supply two camera ball observations.")
    for field in ("camera_id", "frame_id", "calibration_signature"):
        if field in first or field in second:
            if field not in first or field not in second or first[field] != second[field]:
                raise ValueError(f"Ball observations must share the same {field}; transform different cameras first.")
    centers = []
    for observed in (first, second):
        if any(field not in observed for field in ("pixel", "distance_m", "distance_kind", "measurement_reference")):
            raise ValueError("Need each ball's image position, metric distance, distance kind, and surface/center reference.")
        reference = observed["measurement_reference"]
        if reference not in ("surface", "center"):
            raise ValueError("Measurement reference must explicitly be 'surface' or 'center'")
        center = unproject_measurement(observed["pixel"], observed["distance_m"], camera_matrix,
                                       distance_kind=observed["distance_kind"])
        if reference == "surface":
            center = center + radius_m * center / np.linalg.norm(center)
        centers.append(center)
    separation = float(np.linalg.norm(centers[1] - centers[0]) * 100)
    if not np.isfinite(separation):
        raise ValueError("Measurements produce a nonfinite separation; check distance units and calibration.")
    return {"center_distance_cm": separation,
            "surface_gap_cm": separation - 2 * radius_m * 100,
            "camera_centers_m": [center.tolist() for center in centers]}


def _dimensions(width_cm, height_cm, ball_radius_cm, pocket_width_cm, inset_cm):
    values = np.array([width_cm, height_cm, ball_radius_cm, pocket_width_cm, inset_cm], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Table dimensions, ball radius, pocket width and marker inset must be positive")
    width, height, radius, pocket, inset = values
    if 2 * inset >= min(width, height):
        raise ValueError("Marker inset leaves no usable calibration rectangle")
    if 4 * radius >= min(width, height):
        raise ValueError("Ball radius is too large to fit the table")
    # Actual mouth widths are validated by set_pockets/normalize_pockets. A
    # fully overridden default must not invalidate otherwise usable geometry.
    return tuple(float(v) for v in values)


def _marker_geometry(marker_pixels, image_size):
    try:
        points = np.asarray([_pixel(p) for p in marker_pixels], dtype=float)
    except TypeError as exc:
        raise ValueError("Four marker pixels are required in TL, TR, BR, BL order") from exc
    if points.shape != (4, 2):
        raise ValueError("Four marker pixels are required in TL, TR, BR, BL order")
    width, height = image_size
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError("Calibration marker is outside the stereo image")
    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if np.any(crosses <= 1.0) or np.any(np.linalg.norm(edges, axis=1) < 8.0):
        raise ValueError("Markers must form a nondegenerate convex TL, TR, BR, BL polygon")
    if (points[[0, 1], 1].mean() >= points[[2, 3], 1].mean()
            or points[[0, 3], 0].mean() >= points[[1, 2], 0].mean()):
        raise ValueError("Marker order must be TL, TR, BR, BL with table x right and y down")
    return points


@dataclass
class TableCalibration:
    width_cm: float
    height_cm: float
    ball_radius_cm: float
    pocket_width_cm: float
    inset_cm: float
    camera_matrix: np.ndarray
    image_size: tuple
    origin: np.ndarray
    rotation: np.ndarray
    normal_sign: float
    signature: object
    fit_error_cm: float = 0.0
    max_dimension_error_cm: float = 0.0
    marker_pixels: list = field(default_factory=list)
    pockets: list = field(default_factory=list)
    reference_type: str = "markers"

    def __post_init__(self):
        dimensions = _dimensions(self.width_cm, self.height_cm, self.ball_radius_cm,
                                 self.pocket_width_cm, self.inset_cm)
        (self.width_cm, self.height_cm, self.ball_radius_cm,
         self.pocket_width_cm, self.inset_cm) = dimensions
        self.camera_matrix = _camera_matrix(self.camera_matrix)
        self.image_size = _image_size(self.image_size)
        self.origin = np.asarray(self.origin, dtype=float).copy()
        self.rotation = np.asarray(self.rotation, dtype=float).copy()
        if (self.origin.shape != (3,) or self.rotation.shape != (3, 3)
                or not np.isfinite(self.origin).all() or not np.isfinite(self.rotation).all()
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-6)):
            raise ValueError("Table calibration must contain a finite rigid rotation and origin")
        if self.normal_sign not in (-1, 1):
            raise ValueError("Table normal sign must be -1 or +1")
        self.normal_sign = float(self.normal_sign)
        if self.reference_type not in ("markers", "balls"):
            raise ValueError("Calibration reference type must be 'markers' or 'balls'")
        self.signature = _json_copy(self.signature)
        if (not np.isfinite([self.fit_error_cm, self.max_dimension_error_cm]).all()
                or min(self.fit_error_cm, self.max_dimension_error_cm) < 0):
            raise ValueError("Calibration error metrics must be finite and nonnegative")
        if np.dot(self.normal, -self.origin) <= self.ball_radius_cm / 100:
            raise ValueError("Table normal must face the camera, above the ball center plane")
        if self.marker_pixels:
            self.marker_pixels = _marker_geometry(self.marker_pixels, self.image_size).tolist()
        self.set_pockets(self.pockets)
        # Reject malformed loaded profiles, including a table behind the camera.
        for x, y in ((0, 0), (self.width_cm, 0),
                     (self.width_cm, self.height_cm), (0, self.height_cm)):
            self.project(x, y)

    @property
    def normal(self):
        """Unit felt normal directed toward the camera."""
        return self.rotation[:, 2] * self.normal_sign

    @classmethod
    def fit(cls, marker_pixels, depth_map, camera_matrix, image_size, width_cm,
            height_cm, ball_radius_cm, pocket_width_cm, inset_cm, signature,
            max_error_cm=1.0, pockets=None):
        """Fit a metric rigid transform from four same-frame stereo samples."""
        size = _image_size(image_size)
        matrix = _camera_matrix(camera_matrix)
        width, height, radius, pocket, inset = _dimensions(
            width_cm, height_cm, ball_radius_cm, pocket_width_cm, inset_cm)
        if not np.isfinite(max_error_cm) or max_error_cm <= 0:
            raise ValueError("Calibration tolerance must be finite positive centimeters")
        depth = np.asarray(depth_map, dtype=float)
        if depth.shape != (size[1], size[0]):
            raise ValueError("Stereo depth resolution differs from the calibration image size")
        pixels = _marker_geometry(marker_pixels, size)
        observed = np.array([unproject(p, sample_depth(depth, p)["depth_m"], matrix)
                             for p in pixels])
        return cls._fit_observed(pixels, observed, matrix, size, width, height,
                                 radius, pocket, inset, signature, max_error_cm, pockets)

    @classmethod
    def fit_measurements(cls, markers, camera_matrix, image_size, width_cm,
                         height_cm, ball_radius_cm, pocket_width_cm, inset_cm,
                         signature, max_error_cm=1.0, pockets=None):
        """Align using four same-frame pixel/distance readings on flat markers.

        Markers are ordered TL, TR, BR, BL. Measurements refer to the marker
        centers on the felt, so no sphere-radius correction is applied here.
        Distances must already be metric, explicitly camera_z or range.
        """
        size, matrix = _image_size(image_size), _camera_matrix(camera_matrix)
        width, height, radius, pocket, inset = _dimensions(
            width_cm, height_cm, ball_radius_cm, pocket_width_cm, inset_cm)
        tolerance = _measurement_number(max_error_cm, "Calibration tolerance", positive=True)
        if (not isinstance(markers, list) or len(markers) != 4
                or any(not isinstance(m, dict) for m in markers)):
            raise ValueError("Supply four marker measurements in TL, TR, BR, BL order.")
        if [m.get("name") for m in markers] != ["TL", "TR", "BR", "BL"]:
            raise ValueError("Marker names must be TL, TR, BR, BL in that order.")
        for field in ("frame_id", "camera_id"):
            if any(field in m for m in markers):
                values = [m.get(field) for m in markers]
                if (any(isinstance(v, bool) or not isinstance(v, (int, str)) for v in values)
                        or any(v != values[0] for v in values)):
                    raise ValueError(f"All marker measurements must share one {field}.")
        try:
            pixels = _marker_geometry([m["pixel"] for m in markers], size)
            observed = np.array([unproject_measurement(m["pixel"], m["distance_m"], matrix,
                                                       distance_kind=m["distance_kind"])
                                 for m in markers])
        except KeyError as exc:
            raise ValueError("Each marker needs pixel, distance_m and distance_kind.") from exc
        return cls._fit_observed(pixels, observed, matrix, size, width, height,
                                 radius, pocket, inset, signature, tolerance, pockets)

    @classmethod
    def fit_balls(cls, ball_pixels, depth_map, camera_matrix, image_size, width_cm,
                  height_cm, ball_radius_cm, pocket_width_cm, signature,
                  max_error_cm=1.0, pockets=None, cushion_gap_cm=7.62,
                  ball_radii_px=None):
        """Align four resting balls from central visible-surface stereo samples.

        Order the ball centers TL, TR, BR, BL. Each sphere's outside surface is
        ``cushion_gap_cm`` from both adjacent inner cushion lines. The saved
        transform describes the felt, not the plane through the sphere centers.
        """
        size = _image_size(image_size)
        pixels = _marker_geometry(ball_pixels, size)
        depth = np.asarray(depth_map, dtype=float)
        if depth.shape != (size[1], size[0]):
            raise ValueError("Stereo depth resolution differs from the calibration image size")
        if ball_radii_px is None:
            radii = [None] * 4
        else:
            try:
                radii = list(ball_radii_px)
            except TypeError as exc:
                raise ValueError("Supply four ball image radii in TL, TR, BR, BL order") from exc
            if len(radii) != 4:
                raise ValueError("Supply four ball image radii in TL, TR, BR, BL order")
        measurements = []
        for name, pixel, radius_px in zip(("TL", "TR", "BR", "BL"), pixels, radii):
            sampling_radius = 4.0
            if radius_px is not None:
                radius_px = _measurement_number(radius_px, "Ball image radius", positive=True)
                if radius_px < 3:
                    raise ValueError("Ball image radius is too small for reliable stereo sampling")
                sampling_radius = max(1.5, min(4.0, radius_px * 0.22))
            sampled = sample_depth(depth, pixel, radius_px=sampling_radius)
            measured = {"name": name, "pixel": pixel.tolist(), "distance_m": sampled["depth_m"],
                        "distance_kind": "camera_z", "measurement_reference": "surface"}
            if radius_px is not None:
                measured["radius_px"] = radius_px
            measurements.append(measured)
        return cls.fit_ball_measurements(
            measurements, camera_matrix, size, width_cm, height_cm, ball_radius_cm,
            pocket_width_cm, signature, max_error_cm=max_error_cm, pockets=pockets,
            cushion_gap_cm=cushion_gap_cm)

    @classmethod
    def fit_ball_measurements(cls, balls, camera_matrix, image_size, width_cm,
                              height_cm, ball_radius_cm, pocket_width_cm, signature,
                              max_error_cm=1.0, pockets=None, cushion_gap_cm=7.62):
        """Fit the felt pose from four same-frame, measured sphere centers.

        Place exactly four same-size balls near the four corners. The default
        three-inch gap is from the ball surface to each adjacent cushion face;
        the center inset is therefore 7.62 cm plus the measured ball radius.
        Every reading must explicitly identify surface/center and range/camera_z.
        Surface readings receive one radius correction along the camera ray.
        """
        size, matrix = _image_size(image_size), _camera_matrix(camera_matrix)
        gap = _measurement_number(cushion_gap_cm, "Ball-to-cushion surface gap", positive=True)
        radius = _measurement_number(ball_radius_cm, "Ball radius in centimeters", positive=True)
        width, height, radius, pocket, inset = _dimensions(
            width_cm, height_cm, radius, pocket_width_cm, gap + radius)
        tolerance = _measurement_number(max_error_cm, "Calibration tolerance", positive=True)
        if (not isinstance(balls, list) or len(balls) != 4
                or any(not isinstance(ball, dict) for ball in balls)):
            raise ValueError("Supply exactly four ball measurements in TL, TR, BR, BL order.")
        if [ball.get("name") for ball in balls] != ["TL", "TR", "BR", "BL"]:
            raise ValueError("Ball names must be TL, TR, BR, BL in that order.")
        for field in ("frame_id", "camera_id"):
            if any(field in ball for ball in balls):
                values = [ball.get(field) for ball in balls]
                if (any(isinstance(value, bool) or not isinstance(value, (int, str)) for value in values)
                        or any(value != values[0] for value in values)):
                    raise ValueError(f"All calibration balls must share one {field}.")
        if any("calibration_signature" in ball for ball in balls):
            if any("calibration_signature" not in ball
                   or _json_copy(ball["calibration_signature"]) != _json_copy(signature)
                   for ball in balls):
                raise ValueError("All calibration balls must match the camera calibration_signature.")
        try:
            pixels = _marker_geometry([ball["pixel"] for ball in balls], size)
            observed = []
            for ball, pixel in zip(balls, pixels):
                reference = ball["measurement_reference"]
                if reference not in ("surface", "center"):
                    raise ValueError("Measurement reference must explicitly be 'surface' or 'center'")
                center = unproject_measurement(pixel, ball["distance_m"], matrix,
                                               distance_kind=ball["distance_kind"])
                if reference == "surface":
                    center += radius / 100 * center / np.linalg.norm(center)
                if "radius_px" in ball:
                    radius_px = _measurement_number(ball["radius_px"], "Ball image radius", positive=True)
                    if radius_px < 3:
                        raise ValueError("Ball image radius is too small for reliable alignment")
                    if (np.any(pixel - radius_px < 0)
                            or np.any(pixel + radius_px >= np.asarray(size))):
                        raise ValueError("Calibration ball is partially outside the camera image")
                    expected = np.sqrt(matrix[0, 0] * matrix[1, 1]) * radius / 100 / center[2]
                    if not 0.45 <= radius_px / expected <= 2.2:
                        raise ValueError("Calibration ball radius disagrees with its physical size and distance")
                observed.append(center)
        except KeyError as exc:
            raise ValueError("Each calibration ball needs pixel, distance_m, distance_kind, "
                             "and measurement_reference.") from exc
        return cls._fit_observed(
            pixels, np.asarray(observed), matrix, size, width, height, radius, pocket,
            inset, signature, tolerance, pockets, reference_height_cm=radius,
            reference_type="balls")

    @classmethod
    def _fit_observed(cls, pixels, observed, matrix, size, width, height, radius,
                      pocket, inset, signature, max_error_cm, pockets,
                      reference_height_cm=0.0, reference_type="markers"):
        local = np.array([[inset, inset, 0], [width - inset, inset, 0],
                          [width - inset, height - inset, 0], [inset, height - inset, 0]]) / 100
        # Check all edges and diagonals independently of the fitted pose. No
        # scale is optimized: an inaccurate stereo scale must fail calibration.
        pair_errors = [abs(np.linalg.norm(observed[j] - observed[i])
                           - np.linalg.norm(local[j] - local[i])) * 100
                       for i in range(4) for j in range(i + 1, 4)]
        max_dimension_error = float(max(pair_errors))
        if max_dimension_error > max_error_cm:
            raise ValueError("Calibration reference distances disagree with measured table dimensions "
                             f"by {max_dimension_error:.2f} cm; check order, scale and calibration")
        local_centered = local - local.mean(axis=0)
        observed_centered = observed - observed.mean(axis=0)
        u, singular, vt = np.linalg.svd(local_centered.T @ observed_centered)
        if singular[1] < 1e-10:
            raise ValueError("Calibration markers do not span a measurable table plane")
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(vt.T @ u.T)
        rotation = vt.T @ correction @ u.T
        origin = observed.mean(axis=0) - rotation @ local.mean(axis=0)
        fitted = local @ rotation.T + origin
        max_residual = float(np.linalg.norm(fitted - observed, axis=1).max() * 100)
        if max_residual > max_error_cm:
            raise ValueError(f"Table plane fit error {max_residual:.2f} cm exceeds calibration tolerance")
        normal_sign = 1.0 if np.dot(rotation[:, 2], -origin) > 0 else -1.0
        # Fitting balls recovers their center plane. Move the origin one radius
        # away from the camera-facing normal to recover the actual felt plane.
        origin -= rotation[:, 2] * normal_sign * reference_height_cm / 100
        return cls(width, height, radius, pocket, inset, matrix, size, origin,
                   rotation, normal_sign, signature, max_residual,
                   max_dimension_error, pixels.tolist(), [] if pockets is None else pockets,
                   reference_type=reference_type)

    @property
    def table(self):
        table = {"width": self.width_cm, "height": self.height_cm,
                 "ball_radius": self.ball_radius_cm, "pocket_width": self.pocket_width_cm}
        if self.pockets:
            table["pockets"] = _json_copy(self.pockets)
        return table

    def set_pockets(self, pockets):
        """Replace optional measured mouth positions, widths and blocked flags.

        Coordinates are centimeters along the inner cushion rectangle. The six
        IDs are TL, TM, TR, BL, BM and BR. Partial entries inherit the calibrated
        defaults. An empty list restores the standard six-pocket geometry.
        Validation is shared with the planner so a saved profile cannot disagree
        with its pocket-mouth or cushion model. Failed updates leave it intact.
        """
        from billiards import normalize_pockets

        if not isinstance(pockets, list):
            raise ValueError("Pocket overrides must be a list")
        geometry = {"width": self.width_cm, "height": self.height_cm,
                    "ball_radius": self.ball_radius_cm, "pocket_width": self.pocket_width_cm,
                    "pockets": pockets}
        normalized = normalize_pockets(geometry)
        self.pockets = _json_copy(normalized) if pockets else []

    @property
    def polygon(self):
        return np.array([self.project(x, y) for x, y in
                         ((0, 0), (self.width_cm, 0),
                          (self.width_cm, self.height_cm), (0, self.height_cm))])

    def project(self, x_cm, y_cm, height_cm=0):
        """Project table coordinates; positive height is above the felt."""
        if not np.isfinite([x_cm, y_cm, height_cm]).all():
            raise ValueError("Table coordinates must be finite")
        local = np.array([x_cm, y_cm, height_cm * self.normal_sign]) / 100
        camera = self.origin + self.rotation @ local
        if camera[2] <= 0:
            raise ValueError("Projected table point is behind the camera")
        projected = self.camera_matrix @ camera
        return float(projected[0] / projected[2]), float(projected[1] / projected[2])

    def localize_ball_measurement(self, pixel, distance_m, *, distance_kind,
                                  measurement_reference, radius_px=None,
                                  distance_uncertainty_m=0.0, pixel_uncertainty_px=1.0,
                                  max_height_error_cm=1.5):
        """Map a pixel and calibrated camera-to-ball measurement onto the table.

        The pixel must be the projected sphere center in the rectified camera
        described by this calibration. ``measurement_reference`` is ``center``
        when the distance already describes the ball center, or ``surface``
        for its nearest visible surface along that ray. Surface measurements
        receive one ball-radius correction along the ray. A distance alone
        cannot locate a ball: the pixel direction and calibrated camera pose
        are both required. Relative/normalized depth must first be converted
        by a physically validated metric depth provider.

        The result uses table centimeters, with camera XYZ retained in meters.
        The uncertainty envelope is an engineering estimate from the declared
        measurement tolerances and calibration residual, not a confidence bound
        or a substitute for sensor accuracy validation.
        """
        from billiards import ball_center_is_playable

        point = _pixel(pixel)
        distance = _measurement_number(distance_m, "Metric distance in meters", positive=True)
        distance_error = _measurement_number(distance_uncertainty_m, "Distance uncertainty in meters")
        pixel_error = _measurement_number(pixel_uncertainty_px, "Ball pixel uncertainty", positive=True)
        height_tolerance = _measurement_number(max_height_error_cm, "Ball height tolerance")
        if measurement_reference not in ("surface", "center"):
            raise ValueError("Measurement reference must explicitly be 'surface' or 'center'")
        if distance_error >= distance:
            raise ValueError("Distance uncertainty must be smaller than the measured ball distance")
        if not (0 <= point[0] < self.image_size[0] and 0 <= point[1] < self.image_size[1]):
            raise ValueError("Ball pixel is outside the calibrated camera image")
        if radius_px is not None:
            radius_px = _measurement_number(radius_px, "Ball image radius", positive=True)
            if radius_px < 3:
                raise ValueError("Ball image radius is too small for reliable localization")
            if (point[0] - radius_px < 0 or point[1] - radius_px < 0
                    or point[0] + radius_px >= self.image_size[0]
                    or point[1] + radius_px >= self.image_size[1]):
                raise ValueError("Ball is partially outside the calibrated camera image")
        radius_m = self.ball_radius_cm / 100

        def center_from(observed_pixel, observed_distance):
            measured = unproject_measurement(observed_pixel, observed_distance,
                                             self.camera_matrix, distance_kind=distance_kind)
            center = measured.copy()
            if measurement_reference == "surface":
                center += radius_m * measured / np.linalg.norm(measured)
            return measured, center

        measured, center = center_from(point, distance)
        local = self.rotation.T @ (center - self.origin)
        x, y = local[:2] * 100
        height_cm = float(local[2] * self.normal_sign * 100)
        height_error = abs(height_cm - self.ball_radius_cm)
        if height_error > height_tolerance:
            raise ValueError("Measured ball center does not lie at the expected height above the felt")
        if not ball_center_is_playable(self.table, float(x), float(y)):
            raise ValueError("Measured ball center lies outside the playable table bounds")
        expected_radius = float(np.sqrt(self.camera_matrix[0, 0] * self.camera_matrix[1, 1])
                                * radius_m / center[2])
        if radius_px is not None and not 0.45 <= radius_px / expected_radius <= 2.2:
            raise ValueError("Detected ball radius disagrees with its known physical size and distance")

        offsets = []
        for dx in (-pixel_error, pixel_error):
            for dy in (-pixel_error, pixel_error):
                for delta in (-distance_error, distance_error):
                    _, possible_center = center_from(point + [dx, dy], distance + delta)
                    possible_local = self.rotation.T @ (possible_center - self.origin)
                    offsets.append(float(np.linalg.norm(possible_local[:2] - local[:2]) * 100))
        uncertainty = max(offsets) + max(self.fit_error_cm, self.max_dimension_error_cm)
        result = {"x": float(x), "y": float(y), "height_cm": height_cm,
                  "height_error_cm": float(height_error), "depth_m": float(measured[2]),
                  "center_depth_m": float(center[2]), "center_range_m": float(np.linalg.norm(center)),
                  "distance_m": distance, "distance_kind": distance_kind,
                  "measurement_reference": measurement_reference,
                  "distance_uncertainty_m": distance_error, "pixel_uncertainty_px": pixel_error,
                  "camera_xyz_m": center.tolist(), "measured_camera_xyz_m": measured.tolist(),
                  "pixel_space": "rectified", "camera_frame": "calibration_camera",
                  "position_uncertainty_cm": float(uncertainty),
                  "uncertainty_source": "pixel_distance_tolerance_and_calibration_residual",
                  "depth_source": "metric_distance",
                  "localization_source": "calibrated_distance_and_pixel"}
        if radius_px is not None:
            result.update(radius_px=radius_px, expected_radius_px=expected_radius)
        return result

    def localize_ball(self, pixel, radius_px, depth_map, max_height_error_cm=1.5,
                      pixel_uncertainty_px=1.0):
        """Approximate a sphere center using its visible central stereo patch.

        Assumes the supplied pixel is the projected sphere center. A radius is
        added along the viewing ray to correct visible-surface depth. This is
        an approximation, validated against known table height and ball size.
        """
        from billiards import ball_center_is_playable

        point = _pixel(pixel)
        if not np.isfinite(radius_px) or radius_px < 3:
            raise ValueError("Ball image radius is too small for reliable stereo sampling")
        if not np.isfinite(max_height_error_cm) or max_height_error_cm < 0:
            raise ValueError("Ball height tolerance must be finite and nonnegative")
        if not np.isfinite(pixel_uncertainty_px) or pixel_uncertainty_px <= 0:
            raise ValueError("Ball pixel uncertainty must be finite and positive")
        depth = np.asarray(depth_map, dtype=float)
        if depth.shape != (self.image_size[1], self.image_size[0]):
            raise ValueError("Stereo depth resolution differs from table calibration")
        if (point[0] - radius_px < 0 or point[1] - radius_px < 0
                or point[0] + radius_px >= self.image_size[0]
                or point[1] + radius_px >= self.image_size[1]):
            raise ValueError("Ball is partially outside the stereo image")
        sampled = sample_depth(depth, point, radius_px=max(1.5, min(4.0, radius_px * 0.22)))
        surface = unproject(point, sampled["depth_m"], self.camera_matrix)
        radius_m = self.ball_radius_cm / 100
        center = surface + radius_m * surface / np.linalg.norm(surface)
        local = self.rotation.T @ (center - self.origin)
        x, y = local[:2] * 100
        height_cm = float(local[2] * self.normal_sign * 100)
        height_error = abs(height_cm - self.ball_radius_cm)
        if height_error > max_height_error_cm:
            raise ValueError("Stereo ball center does not lie at the expected height above the felt")
        # Balls may hang in a pocket mouth closer than one radius to the outer
        # cushion rectangle. Test the measured openings and actual cushion faces
        # instead of shrinking the entire rectangle and discarding those balls.
        if not ball_center_is_playable(self.table, float(x), float(y)):
            raise ValueError("Stereo ball center lies outside the playable table bounds")
        expected_radius = float(np.sqrt(self.camera_matrix[0, 0] * self.camera_matrix[1, 1])
                                * radius_m / center[2])
        if not 0.45 <= radius_px / expected_radius <= 2.2:
            raise ValueError("Detected ball radius disagrees with its known physical size and stereo depth")
        uncertainty_cm = self._position_uncertainty(
            point, sampled, radius_m, local[:2], pixel_uncertainty_px)
        return {"x": float(x), "y": float(y), **sampled,
                "center_depth_m": float(center[2]), "height_cm": height_cm,
                "center_range_m": float(np.linalg.norm(center)),
                "camera_xyz_m": center.tolist(), "measured_camera_xyz_m": surface.tolist(),
                "pixel_space": "rectified", "camera_frame": "calibration_camera",
                "distance_m": sampled["depth_m"], "distance_kind": "camera_z",
                "measurement_reference": "surface",
                "height_error_cm": float(height_error), "radius_px": float(radius_px),
                "expected_radius_px": expected_radius,
                "position_uncertainty_cm": uncertainty_cm,
                "uncertainty_source": "pixel_depth_spread_and_calibration_residual",
                "depth_source": "calibrated_stereo",
                "localization_source": "sphere_center_from_stereo_patch"}

    def _position_uncertainty(self, point, sampled, radius_m, local_xy,
                              pixel_uncertainty_px):
        """Propagate a measurement envelope into the planner's XY clearance.

        The envelope combines the supplied center-pixel tolerance, the measured
        depth interquartile spread, and the worst calibration residual. This is
        an engineering estimate, not a statistical confidence interval or a bound
        on systematic stereo errors. Calibration must still be checked physically.
        """
        depth_delta = sampled["depth_iqr_m"]
        offsets = []
        for dx in (-pixel_uncertainty_px, pixel_uncertainty_px):
            for dy in (-pixel_uncertainty_px, pixel_uncertainty_px):
                for dz in (-depth_delta, depth_delta):
                    depth_m = sampled["depth_m"] + dz
                    if depth_m <= 0:
                        raise ValueError("Stereo depth spread exceeds the measured ball distance")
                    surface = unproject(point + [dx, dy], depth_m, self.camera_matrix)
                    center = surface + radius_m * surface / np.linalg.norm(surface)
                    offset = (self.rotation.T @ (center - self.origin))[:2] - local_xy
                    offsets.append(float(np.linalg.norm(offset) * 100))
        return float(max(offsets) + max(self.fit_error_cm, self.max_dimension_error_cm))

    def to_dict(self):
        return {"version": 1, "width_cm": self.width_cm, "height_cm": self.height_cm,
                "ball_radius_cm": self.ball_radius_cm, "pocket_width_cm": self.pocket_width_cm,
                "inset_cm": self.inset_cm, "camera_matrix": self.camera_matrix.tolist(),
                "image_size": list(self.image_size), "origin": self.origin.tolist(),
                "rotation": self.rotation.tolist(), "normal_sign": self.normal_sign,
                "signature": _json_copy(self.signature), "fit_error_cm": self.fit_error_cm,
                "max_dimension_error_cm": self.max_dimension_error_cm,
                "marker_pixels": self.marker_pixels, "pockets": _json_copy(self.pockets),
                "reference_type": self.reference_type}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("Unsupported billiards table calibration format")
        fields = ("width_cm", "height_cm", "ball_radius_cm", "pocket_width_cm", "inset_cm",
                  "camera_matrix", "image_size", "origin", "rotation", "normal_sign", "signature",
                  "fit_error_cm", "max_dimension_error_cm", "marker_pixels")
        try:
            return cls(**{key: value[key] for key in fields}, pockets=value.get("pockets", []),
                       reference_type=value.get("reference_type", "markers"))
        except (KeyError, TypeError) as exc:
            raise ValueError("Incomplete or invalid billiards table calibration") from exc

    def save(self, path):
        """Atomically replace a local JSON calibration profile."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=target.parent, prefix=target.name + ".",
                                             suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(self.to_dict(), handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path, signature):
        with Path(path).open(encoding="utf-8") as handle:
            calibration = cls.from_dict(json.load(handle))
        if calibration.signature != _json_copy(signature):
            raise ValueError("Table calibration signature differs from this stereo rig; recalibrate the table")
        return calibration
