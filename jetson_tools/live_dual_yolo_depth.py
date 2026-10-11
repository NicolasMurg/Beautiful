#!/usr/bin/env python3
"""Two-camera local YOLO detections with per-view estimated depth in meters."""

from jetson_paths import PROJECT_ROOT, RUNTIME_ROOT
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

REPO = PROJECT_ROOT
sys.path.insert(0, str(REPO))
from dual_camera_preview import Camera, FIRST, SECOND
from perception_quality import closest_pair, depth_quality, DepthSmoother, appearance_signature
from depth import box_depth
from live_yolo_object_depth import object_depth_records

CHECKPOINT = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'
WINDOW = 'Jieli | Pro Stream - YOLO and estimated depth (Q to exit)'


def distance_band(value):
    """User-selected display range, not a measured confidence interval."""
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return [float(value * .9), float(value * 1.1)]


def distance_text(value):
    band = distance_band(value)
    if band is None:
        return 'unknown'
    return f'{value:.2f}m +/-10% [{band[0]:.2f}-{band[1]:.2f}m]'


def draw_depth_labels(image, records, cv2):
    """Keep the nominal estimate and show its requested +/-10% display band."""
    height, width = image.shape[:2]
    for record in records:
        shown_depth = record.get('depth_smoothed_m', record['depth_m'])
        value = distance_text(shown_depth)
        if shown_depth is None:
            reasons = {'insufficient_stereo_depth': 'too few stereo matches',
                       'mixed_depth_region': 'depth varies too much',
                       'ball_too_small': 'ball too small for stereo'}
            reason = ('stale frame' if record.get('smoothing_status')=='stale_frame'
                      else reasons.get(record.get('measurement_status')))
            if reason: value += f' ({reason})'
        coverage = record.get('valid_depth_fraction')
        quality = f" | matches {coverage:.0%}" if coverage is not None else ''
        if record.get('depth_sample_count') is not None:
            quality += f" ({record['valid_depth_count']}/{record['depth_sample_count']}; need {record['depth_min_valid_count']})"
        if record.get('smoothing_samples',0)>1: quality += ' | smoothed'
        text = f"{record['object']} Z: {value}{quality}"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .5, 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        if tw + 8 > width:
            scale *= max(1, width - 8) / (tw + 8)
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x1, y1, _, y2 = record['bbox_xyxy']
        x = max(0, min(int(x1) + 3, width - tw - 8))
        y = max(th + 6, min(int(y2) - 6, height - baseline - 4))
        if y < y1:
            y = max(th + 6, min(int(y1) + th + 6, height - baseline - 4))
        color = (0,255,255) if distance_band(shown_depth) else (0,165,255)
        cv2.rectangle(image, (x, y-th-4), (min(width-1, x+tw+8), y+baseline+3), (0,0,0), -1)
        cv2.putText(image, text, (x+4,y), font, scale, color, thickness, cv2.LINE_AA)


def right_stereo_depth(left_disparity, right_disparity, geometry, left_roi, right_roi, tolerance=1):
    """Right-view Z with reverse-match consistency; unmatched pixels remain NaN."""
    import numpy as np
    h, w = right_disparity.shape
    yy, xx = np.indices((h, w))
    safe = np.where(np.isfinite(right_disparity), right_disparity, 0)
    lx = np.rint(xx - safe).astype(np.int64)
    x, y, rw, rh = right_roi
    a, b, aw, ah = left_roi
    valid = (np.isfinite(right_disparity) & (right_disparity < 0)
             & (lx >= 0) & (lx < w) & (xx >= x) & (xx < x+rw)
             & (yy >= y) & (yy < y+rh) & (lx >= a) & (lx < a+aw)
             & (yy >= b) & (yy < b+ah))
    reverse = left_disparity[yy, np.clip(lx, 0, w-1)]
    valid &= np.isfinite(reverse) & (np.abs(right_disparity + reverse) <= tolerance)
    effective = -right_disparity - (geometry.cx_left - geometry.cx_right)
    valid &= np.isfinite(effective) & (effective > 0)
    result = np.full((h, w), np.nan, np.float32)
    np.divide(geometry.fx * geometry.baseline_m, effective, out=result, where=valid)
    return result


def prepare_fast_stereo(estimator, scale):
    """Scale matching coordinates and intrinsics together; retain full capture."""
    from stereo_depth import StereoGeometry
    estimator.match_scale = scale
    g = estimator.geometry
    estimator.match_geometry = StereoGeometry(g.fx*scale, g.fy*scale,
        (g.cx_left+.5)*scale-.5, (g.cx_right+.5)*scale-.5, g.baseline_m)
    def roi_scaled(roi):
        x, y, w, h = roi
        x0, y0 = math.ceil(x*scale), math.ceil(y*scale)
        return (x0, y0, max(0, math.floor((x+w)*scale)-x0),
                max(0, math.floor((y+h)*scale)-y0))
    estimator.match_rois = [roi_scaled(estimator.left_roi), roi_scaled(estimator.right_roi)]
    disparities = max(16, math.ceil(estimator._left_matcher.getNumDisparities()*scale/16)*16)
    estimator._right_min = 1-disparities
    for matcher in (estimator._left_matcher, estimator._right_matcher):
        matcher.setNumDisparities(disparities)
        matcher.setSpeckleWindowSize(max(1, round(50*scale*scale)))
        matcher.setSpeckleRange(max(1, round(2*scale)))
    estimator._right_matcher.setMinDisparity(estimator._right_min)


def stereo_maps(estimator, left, right, cv2):
    import numpy as np
    if left.shape[:2][::-1] != estimator.image_size or right.shape[:2][::-1] != estimator.image_size:
        raise ValueError('Capture resolution differs from stereo calibration')
    lrect = cv2.remap(left, *estimator._left_maps, cv2.INTER_LINEAR)
    rrect = cv2.remap(right, *estimator._right_maps, cv2.INTER_LINEAR)
    lg = cv2.cvtColor(lrect, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(rrect, cv2.COLOR_BGR2GRAY)
    scale = getattr(estimator, 'match_scale', 1)
    geometry = getattr(estimator, 'match_geometry', estimator.geometry)
    lroi, rroi = getattr(estimator, 'match_rois', (estimator.left_roi, estimator.right_roi))
    if scale != 1:
        size = tuple(round(n*scale) for n in estimator.image_size)
        lg = cv2.resize(lg, size, interpolation=cv2.INTER_AREA)
        rg = cv2.resize(rg, size, interpolation=cv2.INTER_AREA)
    if hasattr(estimator, 'learned_matcher'):
        size = tuple(round(n*scale) for n in estimator.image_size)
        a = cv2.resize(lrect, size, interpolation=cv2.INTER_AREA) if scale != 1 else lrect
        b = cv2.resize(rrect, size, interpolation=cv2.INTER_AREA) if scale != 1 else rrect
        dl, dr = estimator.learned_matcher.both(a, b)
    else:
        dl = estimator._left_matcher.compute(lg, rg).astype(np.float32) / 16
        dr = estimator._right_matcher.compute(rg, lg).astype(np.float32) / 16
        dr[dr <= estimator._right_min - 1] = np.nan
    from stereo_depth import left_right_validity, disparity_to_depth
    valid = left_right_validity(dl, dr, lroi, rroi, tolerance=scale)
    ldepth = disparity_to_depth(dl, geometry, valid)
    rdepth = right_stereo_depth(dl, dr, geometry, lroi, rroi, tolerance=scale)
    return [lrect, rrect], [ldepth, rdepth]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', default=FIRST, help='physical left input in stereo mode')
    p.add_argument('--second', default=SECOND, help='physical right input in stereo mode')
    p.add_argument('--model', type=Path, default=Path(__file__).parent / 'yolo26n.pt')
    p.add_argument('--device', default='0')
    p.add_argument('--imgsz', type=int, default=320, help='YOLO input size; 640 retains more detail at higher compute cost')
    p.add_argument('--conf', type=float, default=.25, help='detection score cutoff; does not improve classification accuracy')
    p.add_argument('--diagnostic-snapshot', type=Path, help='save first rectified pair and raw depth maps locally for diagnosis')
    p.add_argument('--auto-score', action='store_true', help='automatic estimated solids/stripes pots; R new game, U undo')
    p.add_argument('--pocket-model', type=Path, help='locally trained pocket checkpoint')
    p.add_argument('--detection-only', action='store_true', help='skip depth model and stereo matching; keep calibrated preview geometry')
    p.add_argument('--billiards-only', action='store_true', help='only billiard balls, cue balls and pool cues; boxes with numeric scores')
    p.add_argument('--ball-detail', action='store_true', help='additional YOLO pass over region around detected balls; may reduce analysis FPS')
    p.add_argument('--ball-conf', type=float, default=.20, help='minimum ball score; other classes use --conf')
    p.add_argument('--frames', type=int, default=0, help='processed pairs; 0 runs continuously')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--stereo-calibration', type=Path)
    p.add_argument('--stereo-backend', choices=('sgbm','fast-foundation'), default='sgbm')
    p.add_argument('--baseline-m', type=float, help='measured lens-to-lens spacing in meters')
    p.add_argument('--stereo-scale', type=float, choices=(.5, 1.), default=.5,
                   help='stereo matching resolution: .5 faster, 1 full detail; capture stays calibrated')
    p.add_argument('--pair-gap-ms', type=float, default=30, help='maximum host arrival gap in stereo mode')
    p.add_argument('--pair-age-ms', type=float, default=120, help='fresh frame window for closest-time pairing')
    p.add_argument('--reading-age-ms', type=float, default=300, help='clear analysis display after this source-frame age')
    p.add_argument('--smoothing-window', type=int, choices=(1,2,3,4,5), default=3, help='stationary-object median; 1 disables smoothing')
    p.add_argument('--stereo-notice' , default='', help='display a diagnostic note about current stereo setup')
    p.add_argument('--full-precision', action='store_true', help='disable CUDA half precision')
    p.add_argument('--depth-size', type=int, default=280,
                   help='depth input size, multiple of 14; 280 favors speed, 518 checkpoint default')
    p.add_argument('--offline', action='store_true', help='require cached depth checkpoint')
    a = p.parse_args()
    if not all(math.isfinite(v) and v>0 for v in (a.pair_gap_ms,a.pair_age_ms,a.reading_age_ms)):
        p.error('pair and reading time limits must be finite and positive')
    if a.pair_age_ms>a.reading_age_ms: p.error('pair age must not exceed reading age')
    if a.depth_size < 140 or a.depth_size % 14: p.error('--depth-size must be a multiple of 14, at least 140')
    if a.frames < 0 or not a.model.is_file(): p.error('existing local YOLO weights and nonnegative --frames required')
    if a.device != 'cpu' and not a.device.isdecimal(): p.error('--device must be cpu or a CUDA index')
    if os.path.realpath(a.first) == os.path.realpath(a.second): p.error('choose distinct capture devices')
    if a.stereo_calibration and (not a.stereo_calibration.is_file() or a.baseline_m is None
                                or not 0 < a.baseline_m < float('inf')):
        p.error('stereo mode requires an existing calibration and finite positive --baseline-m')
    if a.stereo_backend == 'fast-foundation' and not a.detection_only and (not a.stereo_calibration or a.device == 'cpu'):
        p.error('fast-foundation requires stereo calibration and CUDA')
    if a.imgsz < 160 or a.imgsz > 1280 or a.imgsz % 32:
        p.error('--imgsz must be a multiple of 32 from 160 to 1280')
    if not math.isfinite(a.ball_conf) or not 0 < a.ball_conf < 1:
        p.error('--ball-conf must be between 0 and 1')
    if not math.isfinite(a.conf) or not 0 < a.conf < 1:
        p.error('--conf must be between 0 and 1')
    if a.auto_score and (not a.pocket_model or not a.pocket_model.is_file()):
        p.error('--auto-score requires an existing --pocket-model checkpoint')
    return a


def main(args):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; request --device cpu explicitly')
    cv2.setNumThreads(4)
    torch.set_num_threads(2)
    device = torch.device('cpu' if args.device == 'cpu' else f'cuda:{args.device}')
    print(f'Loading local YOLO: {args.model}', flush=True)
    yolo = YOLO(str(args.model))
    estimator = processor = depth_model = None
    from pool_score import AutoScore, appearance_group
    from pocket_locations import detect_pockets, StablePockets
    score_engine = AutoScore(Path(__file__).parent / 'pool_score.json') if args.auto_score else None
    pocket_model = YOLO(str(args.pocket_model)) if args.auto_score else None
    stable_pockets = StablePockets()
    score_reset, score_undo = threading.Event(), threading.Event()
    pockets, last_pocket_run, previous_score_frame = [], -float('inf'), None
    score_filter_revision = None
    width, height = 640, 480
    source = 'monocular_metric_estimate'
    if args.stereo_calibration:
        from stereo_depth import StereoDepthEstimator
        estimator = StereoDepthEstimator(args.stereo_calibration, args.baseline_m)
        width, height = estimator.image_size
        prepare_fast_stereo(estimator, args.stereo_scale)
        source = 'calibrated_stereo_estimate'
        if args.stereo_backend == 'fast-foundation' and not args.detection_only:
            from fast_stereo_backend import FastStereoMatcher
            estimator.learned_matcher = FastStereoMatcher(args.device, estimator._left_matcher.getNumDisparities())
            source = 'fast_foundation_calibrated_stereo_estimate'
        with np.load(args.stereo_calibration, allow_pickle=False) as calibration:
            if 'standard_alignment_passed' in calibration and not bool(calibration['standard_alignment_passed']):
                source = 'stereo_relaxed_alignment_estimate'
                print('Calibration used relaxed alignment; physical distance accuracy is unvalidated.', flush=True)

        print('Stereo selected: first must be physical left; second physical right.', flush=True)
    elif not args.detection_only:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print(f'Loading metric depth: {CHECKPOINT}', flush=True)
        processor = AutoImageProcessor.from_pretrained(CHECKPOINT, use_fast=False,
                                                       local_files_only=args.offline, size={'height':args.depth_size,'width':args.depth_size})
        depth_model = AutoModelForDepthEstimation.from_pretrained(
            CHECKPOINT, use_safetensors=True, local_files_only=args.offline).to(device).eval()
        if depth_model.config.depth_estimation_type != 'metric':
            raise RuntimeError('Depth checkpoint is not metric; cannot display meters')
        print('Independent per-camera estimated Z in meters; stereo calibration is not active.', flush=True)
    if args.detection_only:
        source = 'detection_only_no_depth'
        print('Detection-only: depth models/matching disabled; no metric distances produced.', flush=True)
    if args.ball_detail:
        print('Ball detail pass enabled: adaptive close view around ball proposals, central-table fallback; additional inference may reduce FPS.', flush=True)
    empty_depth = np.full((height, width), np.nan, dtype=np.float32) if args.detection_only else None
    stop = threading.Event()
    cameras = [Camera(cv2, name, path, 'MJPG', width, height, fps, stop, history_size=8)
               for name, path, fps in [('Jieli', args.first, 25), ('Pro Stream', args.second, 30)]]
    state = {'output': None, 'error': None, 'count': 0, 'analysis_fps': 0}
    smoothers = [DepthSmoother(args.smoothing_window,args.reading_age_ms/1000) for _ in cameras]
    center_smoothers = [DepthSmoother(args.smoothing_window,args.reading_age_ms/1000) for _ in cameras]
    analysis_times = deque(maxlen=30)
    lock = threading.Lock()
    from detection_filters import DetectionFilters, FilterPanel, filter_result, filter_scores, BALL_CATEGORIES
    from ball_perception import ball_depth, detail_box_records, draw_detail_boxes, detail_region
    from diamond_shapes import diamond_records, draw_diamonds
    from billiards_display import billiards_names, draw_score_boxes
    diamond_category = max(yolo.names) + 1
    filter_names = dict(yolo.names)
    filter_names[diamond_category] = 'diamond shape'
    if args.billiards_only:
        filter_names = billiards_names(filter_names)
        if not filter_names:
            raise RuntimeError('Billiards-only mode requires a checkpoint with billiard ball/cue ball/pool cue labels')
    filter_file = 'billiards_detection_filters.json' if args.billiards_only else 'detection_filters.json'
    filters = DetectionFilters(filter_names, Path(__file__).parent / filter_file)
    panel = None

    def filters_changed():
        with lock:
            state['output'] = None


    def infer():
        nonlocal pockets, last_pocket_run, previous_score_frame, score_filter_revision
        previous = [0, 0]
        try:
            while not stop.is_set():
                latest = [c.snapshot() for c in cameras]
                if any(s[0] is None or s[4] is not None or s[5] is not None for s in latest):
                    for smoother in smoothers+center_smoothers: smoother.clear()
                    stop.wait(.01); continue
                if estimator:
                    snaps = closest_pair([c.history_snapshot() for c in cameras],previous,
                                         time.monotonic(),args.pair_gap_ms/1000,args.pair_age_ms/1000)
                    if snaps is None:
                        stop.wait(.005); continue
                else:
                    snaps = latest
                    if any(s[1]<=n for s,n in zip(snaps,previous)):
                        stop.wait(.005); continue
                processing_started = time.monotonic()
                previous = [s[1] for s in snaps]
                frames = [s[0] for s in snaps]
                arrivals = [s[3] for s in snaps]
                pair_gap_ms = abs(arrivals[0]-arrivals[1])*1000
                latest_gap_ms = abs(latest[0][3]-latest[1][3])*1000
                if args.detection_only:
                    if estimator:
                        frames = [cv2.remap(frame, *rect_maps, cv2.INTER_LINEAR) for frame,rect_maps in zip(frames,(estimator._left_maps,estimator._right_maps))]
                    maps = [empty_depth, empty_depth]
                elif estimator:
                    frames, maps = stereo_maps(estimator, *frames, cv2)
                else:
                    maps = []
                    for frame in frames:
                        inputs = processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors='pt')
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.inference_mode():
                            out = depth_model(**inputs).predicted_depth
                            z = torch.nn.functional.interpolate(out.unsqueeze(1), size=frame.shape[:2],
                                      mode='bilinear', align_corners=False)[0, 0].float().cpu().numpy()
                        maps.append(z)
                if args.diagnostic_snapshot and state['count']==0:
                    np.savez_compressed(args.diagnostic_snapshot,first_frame=frames[0],second_frame=frames[1],first_depth=maps[0],second_depth=maps[1])
                depth_finished = time.monotonic()
                if state['count'] == 0:
                    print('Depth maps ready for both cameras; starting YOLO warmup.', flush=True)
                results = yolo.predict(frames, device=args.device, imgsz=args.imgsz, conf=min(args.conf, args.ball_conf), verbose=False,
                                       quantize=16 if args.device != 'cpu' and not args.full_precision else None)
                yolo_finished = time.monotonic()
                enabled, filter_revision = filters.snapshot()
                results = [filter_result(filter_scores(result, args.conf, args.ball_conf), enabled) for result in results]
                detail_results = [None, None]
                offsets = [(0,0), (0,0)]
                if args.ball_detail and any(filters.names[k] in BALL_CATEGORIES for k in enabled):
                    crops=[]
                    for frame,result in zip(frames,results):
                        x1,y1,x2,y2=detail_region(frame.shape,result)
                        crops.append(frame[y1:y2,x1:x2])
                        offsets[len(crops)-1]=(x1,y1)
                    detail_results=yolo.predict(crops,device=args.device,imgsz=args.imgsz,conf=args.ball_conf,verbose=False,
                                                quantize=16 if args.device!='cpu' and not args.full_precision else None)
                yolo_finished = time.monotonic()
                views, records = [], []
                with lock: pair_id = state['count'] + 1
                for c, frame, z, result, snap, smoother, center_smoother in zip(cameras, frames, maps, results, snaps, smoothers, center_smoothers):
                    recs = object_depth_records(result, z, snap[1])
                    for rec, score in zip(recs, result.boxes.conf.cpu().tolist()):
                        rec['detection_confidence'] = float(score)
                        rec['detection_method'] = 'yolo'
                    index=cameras.index(c)
                    if detail_results[index] is not None:
                        recs.extend(detail_box_records(detail_results[index],offsets[index],recs,enabled,z,snap[1],args.ball_conf))
                    if diamond_category in enabled:
                        recs.extend(diamond_records(frame, snap[1], z))
                    if estimator and not args.detection_only:
                        from stereo_depth import object_dimensions
                        for rec in recs:
                            sample_box = [v*args.stereo_scale for v in rec['bbox_xyxy']]
                            if rec['object'] in BALL_CATEGORIES:
                                rec.update(ball_depth(z,sample_box,estimator.match_geometry))
                            else:
                                rec.update(object_dimensions(z, sample_box, estimator.match_geometry))
                                rec.update(depth_quality(z,sample_box))
                    if args.detection_only:
                        for rec in recs:
                            rec.update(depth_m=None,measurement_status='depth_disabled',quality_status='depth_disabled')
                    smoother.update(recs, snap[3], time.monotonic(),
                                    [appearance_signature(frame,r['bbox_xyxy'],cv2) for r in recs])
                    for rec in recs:
                        rec.update(camera=c.name, depth_source=source, depth_enabled=not args.detection_only,
                                   depth_display_band_m=distance_band(rec['depth_m']),
                                   depth_smoothed_display_band_m=distance_band(rec['depth_smoothed_m']),
                                   source_age_ms=(time.monotonic()-snap[3])*1000,
                                   display_band_percent=10)
                    if args.billiards_only:
                        image = frame.copy()
                        draw_score_boxes(image,recs,cv2)
                    else:
                        image = result.plot(masks=False)
                        draw_diamonds(image, recs)
                        draw_detail_boxes(image, recs, cv2)
                        draw_depth_labels(image, recs, cv2)
                    h, w = z.shape
                    radius = 15 * (args.stereo_scale if estimator else 1)
                    center = box_depth(z, [w/2-radius,h/2-radius,w/2+radius,h/2+radius], inner=1)
                    center_box=[w/2-radius,h/2-radius,w/2+radius,h/2+radius]
                    center_quality=depth_quality(z,center_box,inner=1,min_valid=30 if estimator else 20)
                    if estimator and center_quality['quality_status']!='valid_depth': center=None
                    source_fresh=time.monotonic()-snap[3]<=args.reading_age_ms/1000
                    fh,fw=frame.shape[:2]
                    center_record=dict(object='center',depth_m=center,
                                       bbox_xyxy=[fw/2-15,fh/2-15,fw/2+15,fh/2+15])
                    center_smoother.update([center_record],snap[3],time.monotonic(),
                        [appearance_signature(frame,center_record['bbox_xyxy'],cv2)])
                    center_smoothed=center_record['depth_smoothed_m']
                    value = distance_text(center_smoothed) if source_fresh else 'unknown (stale frame)'
                    if center_record['smoothing_samples']>1: value+=' smoothed'
                    if not args.billiards_only:
                        cv2.putText(image, f'{c.name} | center Z: {value}', (8, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1, cv2.LINE_AA)
                    valid = np.isfinite(z) & (z>0)
                    coverage=float(np.mean(valid))
                    if not args.billiards_only:
                        cv2.putText(image,f'Depth coverage {coverage:.0%} | pair gap {pair_gap_ms:.1f}ms',
                                    (8,44),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
                    scaled = (np.clip(np.where(valid,z,0)/3,0,1)*255).astype(np.uint8)
                    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
                    heat[~valid] = 0
                    cv2.putText(heat, f'{c.name} depth | color scale 0-3m; labels use raw Z', (8,22),
                                cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                    if not args.billiards_only:
                        cv2.drawMarker(image,(image.shape[1]//2,image.shape[0]//2),(0,255,255),cv2.MARKER_CROSS,15,1)
                    views.append((image, heat))
                    records.append(dict(camera=c.name, frame_id=snap[1], capture_monotonic_s=snap[3],
                                        center_depth_m=center, center_display_band_m=distance_band(center),
                                        center_depth_smoothed_m=center_smoothed,
                                        center_smoothed_display_band_m=distance_band(center_smoothed),
                                        center_smoothing_status=center_record['smoothing_status'],
                                        center_smoothing_samples=center_record['smoothing_samples'],
                                        display_band_percent=10, center_quality=center_quality,
                                        source_fresh=source_fresh,
                                        capture_fps=(snap[1]-1)/(snap[3]-snap[2]) if snap[3]>snap[2] else 0,
                                        valid_depth_fraction=float(np.mean(valid)), objects=recs))
                score_events = []
                if score_engine is not None:
                    score_now=snaps[0][3]
                    if score_filter_revision != filter_revision:
                        score_engine.tracks.clear()
                        score_filter_revision = filter_revision
                    if score_reset.is_set():score_engine.reset();score_reset.clear()
                    if score_undo.is_set():score_engine.undo();score_undo.clear()
                    if time.monotonic()-last_pocket_run>2:
                        pockets=stable_pockets.update(detect_pockets(pocket_model,frames[0],args.device,not args.full_precision))
                        last_pocket_run=time.monotonic()
                    gray=cv2.resize(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),(160,120)).astype(np.float32)
                    scene_clear=previous_score_frame is None or np.mean(np.abs(gray-previous_score_frame)>25)<.18
                    previous_score_frame=gray
                    observations=[]
                    for rec in records[0]['objects']:
                        x1,y1,x2,y2=rec['bbox_xyxy']
                        group=appearance_group(frames[0],rec['bbox_xyxy'])
                        rec['group_estimate']=group
                        observations.append(dict(center=[(x1+x2)/2,(y1+y2)/2],radius=min(x2-x1,y2-y1)/2,group=group))
                    scene_clear = (scene_clear and bool(enabled)
                                   and records[0]['source_fresh']
                                   and (time.monotonic()-snaps[0][3])*1000 <= args.reading_age_ms
                                   and filters.snapshot()[1] == filter_revision)
                    score_events=score_engine.update(observations,pockets,score_now,scene_clear)
                    totals=score_engine.totals
                    caption=f"AUTO estimates | Solids {totals['solids']}/7 | Stripes {totals['stripes']}/7 | Scratch {totals['scratch']} | 8-ball {totals['eight']}"
                    for image,_ in views:
                        cv2.putText(image,caption,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,255),1,cv2.LINE_AA)
                        cv2.putText(image,f'Pockets {len(pockets)} | R new game | U undo',(8,42),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1,cv2.LINE_AA)
                analysis_times.append(time.monotonic())
                analysis_fps = ((len(analysis_times)-1)/(analysis_times[-1]-analysis_times[0])
                                if len(analysis_times)>1 else 0)
                if filters.snapshot()[1] != filter_revision:
                    continue  # checkbox changed while these views were rendered
                print(json.dumps(dict(enabled_categories=[filters.names[k] for k in sorted(enabled)], pair_gap_ms=pair_gap_ms, latest_pair_gap_ms=latest_gap_ms,
                                      pairing_time_basis="host_arrival_not_exposure",
                                      analysis_fps=analysis_fps, stereo_scale=args.stereo_scale if estimator else None, pair_id=pair_id,depth_source=source,processing_ms=(time.monotonic()-processing_started)*1000,
                                      depth_ms=(depth_finished-processing_started)*1000,
                                      yolo_ms=(yolo_finished-depth_finished)*1000,scoreboard=dict(score_engine.totals) if score_engine else None,score_events=score_events,pockets=pockets if score_engine else None,cameras=records),allow_nan=False),flush=True)
                with lock:
                    if filters.snapshot()[1] != filter_revision: continue
                    state.update(output=(views, arrivals), count=pair_id, analysis_fps=analysis_fps)
                if args.frames and pair_id >= args.frames: return
        except Exception as exc:
            with lock: state['error'] = str(exc)

    worker = threading.Thread(target=infer, daemon=True, name='perception')
    shown = False
    preview_cache = [None, None]
    preview_ids = [-1, -1]
    capture_rates = [deque(), deque()]
    try:
        if not args.no_display:
            panel = FilterPanel(filters, filters_changed, stop.set)
        for c in cameras: c.thread.start()
        worker.start()
        started = time.monotonic()
        if not args.no_display:
            cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1280,960);shown=True
            panel.root.after(500, panel.show)
        while True:
            if panel:
                panel.poll()
                if stop.is_set(): break
            now = time.monotonic()
            snaps = [c.snapshot() for c in cameras]
            for c,s in zip(cameras,snaps):
                if s[4]: raise RuntimeError(f'{c.name}: {s[4]}')
                if s[5] is not None and now-s[5] >= 8: raise RuntimeError(f'{c.name}: reconnect timeout')
                if s[3] is not None and now-s[3]>3: raise RuntimeError(f'{c.name}: capture is stale')
                if s[1] == 0 and now-started>12: raise RuntimeError(f'{c.name}: startup timeout')
            with lock: output,error,count,analysis_fps = state['output'],state['error'],state['count'],state['analysis_fps']
            if error: raise RuntimeError(error)
            if not args.no_display:
                top,bottom=[],[]
                for i,c in enumerate(cameras):
                    snap = snaps[i]
                    rates = capture_rates[i]
                    rates.append((now, snap[1]))
                    while len(rates)>2 and now-rates[0][0]>2: rates.popleft()
                    fps = (rates[-1][1]-rates[0][1])/max(.001,rates[-1][0]-rates[0][0])
                    if snap[0] is None:
                        image=np.zeros((480,640,3),np.uint8)
                        text=f'{c.name}: '+('reconnecting' if snap[5] else 'waiting for camera')
                        cv2.putText(image,text,(20,240),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                    else:
                        if preview_ids[i] != snap[1]:
                            raw = snap[0]
                            if estimator:
                                maps = estimator._left_maps if i==0 else estimator._right_maps
                                raw = cv2.remap(raw, *maps, cv2.INTER_LINEAR)
                            preview_cache[i] = cv2.resize(raw,(640,480))
                            preview_ids[i] = snap[1]
                        image = preview_cache[i].copy()
                        cv2.putText(image,f'{c.name} LIVE | capture {fps:.1f} FPS | frame {snap[1]}',
                                    (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1,cv2.LINE_AA)
                    if output is None or snap[0] is None:
                        heat=np.zeros((480,640,3),np.uint8)
                        cv2.putText(heat,'Waiting for analysis' if snap[0] is not None else 'Feed unavailable',
                                    (20,240),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                    else:
                        # Analysis boxes stay on their source frame, never on newer live frames.
                        heat=cv2.resize(output[0][i][0],(640,480))
                        inset=cv2.resize(output[0][i][1],(160,120))
                        if not args.billiards_only:
                            heat[330:450,480:640]=inset
                        age=now-output[1][i]
                        if age>args.reading_age_ms/1000:
                            heat[:]=0
                            cv2.putText(heat,'Analysis stale' if args.billiards_only else 'Analysis stale - distances cleared',(20,240),
                                        cv2.FONT_HERSHEY_SIMPLEX,.6,(0,165,255),1,cv2.LINE_AA)
                        cv2.putText(heat,f'ANALYSIS {analysis_fps:.1f} FPS | age {age:.2f}s | {'billiards' if args.billiards_only else 'depth inset'}',
                                    (8,465),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                    if args.stereo_notice:
                        cv2.rectangle(heat,(0,430),(479,453),(0,0,0),-1)
                        cv2.putText(heat,args.stereo_notice,(8,447),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,165,255),1,cv2.LINE_AA)
                    top.append(image);bottom.append(heat)
                cv2.imshow(WINDOW,np.vstack((np.hstack(top),np.hstack(bottom))))
                key=cv2.waitKey(1)&255
                if key in (ord('q'),ord('Q'),27):break
                if score_engine is not None and key in (ord('r'),ord('R')):score_reset.set()
                if score_engine is not None and key in (ord('u'),ord('U')):score_undo.set()
                try:
                    if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1: break
                except cv2.error as exc:
                    if exc.code == -27: break  # Qt removed the final window on close.
                    raise
            if args.frames and count>=args.frames: break
            if not worker.is_alive():
                with lock: final_count,final_error=state['count'],state['error']
                if args.frames and final_count>=args.frames and final_error is None: break
                raise RuntimeError(final_error or 'Inference worker stopped unexpectedly')
            stop.wait(.005)
    finally:
        stop.set()
        for c in cameras: c.thread.join(12)
        worker.join(30)
        if panel: panel.close()
        if shown: cv2.destroyAllWindows()
        print('Stopped; camera cleanup requested.',flush=True)


if __name__=='__main__':
    try: main(parse_args())
    except KeyboardInterrupt: print('Stopped by user.',flush=True)
    except Exception as exc: raise SystemExit(f'Dual YOLO depth failed: {exc}')
