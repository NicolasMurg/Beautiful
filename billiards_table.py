"""Metric billiards table coordinates derived from calibrated stereo depth.

Four textured markers lie on the felt, inset from the inner cushion corners.
Their measured stereo positions fit a rigid transform, never an adjustable
scale or a depth-free homography. Coordinates exposed to the planner are cm.
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
    if (not np.isfinite(radius_px) or radius_px <= 0 or min_valid < 1
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


def _dimensions(width_cm, height_cm, ball_radius_cm, pocket_width_cm, inset_cm):
    values = np.array([width_cm, height_cm, ball_radius_cm, pocket_width_cm, inset_cm], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Table dimensions, ball radius, pocket width and marker inset must be positive")
    width, height, radius, pocket, inset = values
    if 2 * inset >= min(width, height):
        raise ValueError("Marker inset leaves no usable calibration rectangle")
    if 2 * radius >= min(width, height) or not 2 * radius < pocket < min(width, height):
        raise ValueError("Ball diameter and pocket width must fit the table")
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
        self.signature = _json_copy(self.signature)
        if (not np.isfinite([self.fit_error_cm, self.max_dimension_error_cm]).all()
                or min(self.fit_error_cm, self.max_dimension_error_cm) < 0):
            raise ValueError("Calibration error metrics must be finite and nonnegative")
        if np.dot(self.normal, -self.origin) <= self.ball_radius_cm / 100:
            raise ValueError("Table normal must face the camera, above the ball center plane")
        if self.marker_pixels:
            self.marker_pixels = _marker_geometry(self.marker_pixels, self.image_size).tolist()
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
            max_error_cm=1.0):
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
        local = np.array([[inset, inset, 0], [width - inset, inset, 0],
                          [width - inset, height - inset, 0], [inset, height - inset, 0]]) / 100
        observed = np.array([unproject(p, sample_depth(depth, p)["depth_m"], matrix)
                             for p in pixels])
        # Check all edges and diagonals independently of the fitted pose. No
        # scale is optimized: an inaccurate stereo scale must fail calibration.
        pair_errors = [abs(np.linalg.norm(observed[j] - observed[i])
                           - np.linalg.norm(local[j] - local[i])) * 100
                       for i in range(4) for j in range(i + 1, 4)]
        max_dimension_error = float(max(pair_errors))
        if max_dimension_error > max_error_cm:
            raise ValueError("Stereo marker distances disagree with measured table dimensions "
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
        return cls(width, height, radius, pocket, inset, matrix, size, origin,
                   rotation, normal_sign, signature, max_residual,
                   max_dimension_error, pixels.tolist())

    @property
    def table(self):
        return {"width": self.width_cm, "height": self.height_cm,
                "ball_radius": self.ball_radius_cm, "pocket_width": self.pocket_width_cm}

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

    def localize_ball(self, pixel, radius_px, depth_map, max_height_error_cm=1.5):
        """Approximate a sphere center using its visible central stereo patch.

        Assumes the supplied pixel is the projected sphere center. A radius is
        added along the viewing ray to correct visible-surface depth. This is
        an approximation, validated against known table height and ball size.
        """
        point = _pixel(pixel)
        if not np.isfinite(radius_px) or radius_px < 3:
            raise ValueError("Ball image radius is too small for reliable stereo sampling")
        if not np.isfinite(max_height_error_cm) or max_height_error_cm < 0:
            raise ValueError("Ball height tolerance must be finite and nonnegative")
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
        margin = self.ball_radius_cm
        if not (margin - 1e-6 <= x <= self.width_cm - margin + 1e-6
                and margin - 1e-6 <= y <= self.height_cm - margin + 1e-6):
            raise ValueError("Stereo ball center lies outside the playable table bounds")
        expected_radius = float(np.sqrt(self.camera_matrix[0, 0] * self.camera_matrix[1, 1])
                                * radius_m / center[2])
        if not 0.45 <= radius_px / expected_radius <= 2.2:
            raise ValueError("Detected ball radius disagrees with its known physical size and stereo depth")
        return {"x": float(x), "y": float(y), **sampled,
                "center_depth_m": float(center[2]), "height_cm": height_cm,
                "height_error_cm": float(height_error), "radius_px": float(radius_px),
                "expected_radius_px": expected_radius,
                "depth_source": "calibrated_stereo",
                "localization_source": "sphere_center_from_stereo_patch"}

    def to_dict(self):
        return {"version": 1, "width_cm": self.width_cm, "height_cm": self.height_cm,
                "ball_radius_cm": self.ball_radius_cm, "pocket_width_cm": self.pocket_width_cm,
                "inset_cm": self.inset_cm, "camera_matrix": self.camera_matrix.tolist(),
                "image_size": list(self.image_size), "origin": self.origin.tolist(),
                "rotation": self.rotation.tolist(), "normal_sign": self.normal_sign,
                "signature": _json_copy(self.signature), "fit_error_cm": self.fit_error_cm,
                "max_dimension_error_cm": self.max_dimension_error_cm,
                "marker_pixels": self.marker_pixels}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("Unsupported billiards table calibration format")
        fields = ("width_cm", "height_cm", "ball_radius_cm", "pocket_width_cm", "inset_cm",
                  "camera_matrix", "image_size", "origin", "rotation", "normal_sign", "signature",
                  "fit_error_cm", "max_dimension_error_cm", "marker_pixels")
        try:
            return cls(**{key: value[key] for key in fields})
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
