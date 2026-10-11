"""File boundaries shared by the standalone physical-table scripts."""

import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / ".local" / "billiards"


def _constant(value):
    raise ValueError(f"JSON numbers must be finite, not {value}.")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON repeats field {key!r}.")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=_constant, object_pairs_hook=_object)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path, content):
    """Replace one artifact only after its complete content is available."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path, value):
    atomic_write(path, json_bytes(value))


def distinct_outputs(inputs, outputs):
    """Prevent a command from replacing its own calibration or source data."""
    protected = [Path(path).resolve() for path in inputs if path is not None]
    targets = [Path(path).resolve() for path in outputs if path is not None]
    for index, target in enumerate(targets):
        for other in protected + targets[:index]:
            if target == other or (target.exists() and other.exists() and target.samefile(other)):
                raise ValueError("Output paths must differ from all input paths and each other.")
        if target.is_dir():
            raise ValueError(f"Output path is a directory: {target}")


def validate_pixel_frame(value):
    if not isinstance(value, dict):
        raise ValueError("Measurements must be a JSON object.")
    if value.get("pixel_space") != "rectified":
        raise ValueError("pixel_space must be 'rectified'; pixels must match the calibrated camera.")
    size = value.get("image_size")
    if (not isinstance(size, list) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)):
        raise ValueError("image_size must be [width, height] in positive integer pixels.")
    signature = value.get("calibration_signature")
    if not isinstance(signature, (str, dict)) or not signature:
        raise ValueError("Supply calibration_signature from the measurement provider's camera calibration.")
    return tuple(size)


def png_bytes(image):
    import cv2
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the review image as PNG.")
    return encoded.tobytes()
