import cv2
import numpy as np
from diamond_shapes import detect_diamonds, diamond_records


def canvas():
    return np.zeros((300, 400, 3), dtype=np.uint8)


def test_filled_and_outline_diamond_deduplicated():
    for thickness in (-1, 4):
        image = canvas()
        points = np.array([[200, 40], [280, 150], [200, 260], [120, 150]])
        cv2.polylines(image, [points], True, (255,255,255), thickness) if thickness > 0 else cv2.fillPoly(image, [points], (255,255,255))
        detections = detect_diamonds(image)
        assert len(detections) == 1
        x1,y1,x2,y2 = detections[0]['bbox_xyxy']
        assert abs(x1-120) < 6 and abs(x2-280) < 6
        records = diamond_records(image, 4, np.ones((300,400)))
        assert records[0]['detection_confidence'] is None
        assert records[0]['detection_method'] == 'shape_geometry'
        assert records[0]['depth_m'] == 1


def test_non_diamonds_rejected():
    for shape in ('blank', 'circle', 'rectangle', 'triangle', 'tiny'):
        image=canvas()
        if shape=='circle': cv2.circle(image,(200,150),75,(255,255,255),-1)
        if shape=='rectangle': cv2.rectangle(image,(100,70),(300,230),(255,255,255),-1)
        if shape=='triangle': cv2.fillPoly(image,[np.array([[200,40],[280,240],[120,240]])],(255,255,255))
        if shape=='tiny': cv2.fillPoly(image,[np.array([[200,145],[205,150],[200,155],[195,150]])],(255,255,255))
        assert detect_diamonds(image)==[], shape
