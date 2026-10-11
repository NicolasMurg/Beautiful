"""Detect visible upright rhombus outlines; geometry scores are not probabilities."""
import cv2
import numpy as np


def detect_diamonds(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon[:, 0].astype(float)
        sides = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
        if sides.min() / sides.max() < 0.72:
            continue
        # Opposite corners must be top/bottom and left/right, rather than a box.
        top, bottom = points[np.argmin(points[:, 1])], points[np.argmax(points[:, 1])]
        left, right = points[np.argmin(points[:, 0])], points[np.argmax(points[:, 0])]
        if len({tuple(p) for p in (top, bottom, left, right)}) != 4:
            continue
        vertical, horizontal = bottom - top, right - left
        if vertical[1] < 20 or horizontal[0] < 20:
            continue
        if abs(vertical[0]) / vertical[1] > 0.35 or abs(horizontal[1]) / horizontal[0] > 0.35:
            continue
        x, y, w, h = cv2.boundingRect(polygon)
        fill = area / (w * h)
        if not 0.36 < fill < 0.65:
            continue
        midpoint_error = np.linalg.norm((top + bottom - left - right) / 2) / max(w, h)
        if midpoint_error > 0.15:
            continue
        quality = float((sides.min() / sides.max()) * (1 - midpoint_error))
        candidates.append({'bbox_xyxy': [x, y, x + w, y + h], 'shape_quality_score': quality})
    # Inner and outer edges of a drawn outline represent the same pattern.
    selected = []
    for candidate in sorted(candidates, key=lambda c: c['shape_quality_score'], reverse=True):
        a = candidate['bbox_xyxy']
        duplicate = False
        for kept in selected:
            b = kept['bbox_xyxy']
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
            if overlap / smaller > 0.75:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return selected


def diamond_records(frame, frame_id, depth_map):
    from depth import box_depth
    return [dict(candidate, frame_id=frame_id, track_id=None, object='diamond shape',
                 detection_method='shape_geometry', detection_confidence=None,
                 depth_m=box_depth(depth_map, candidate['bbox_xyxy'], inner=0.4))
            for candidate in detect_diamonds(frame)]


def draw_diamonds(image, records):
    for record in records:
        if record.get('detection_method') != 'shape_geometry':
            continue
        x1, y1, x2, y2 = map(int, record['bbox_xyxy'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(image, 'diamond shape (geometry)', (x1, max(16, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
