"""Small-ball stereo sampling and box merging; invalid matches stay unknown."""
import math
import numpy as np


def ball_depth(depth, box, geometry):
    x1,y1,x2,y2=map(float,box)
    h,w=depth.shape
    cx,cy=(x1+x2)/2,(y1+y2)/2
    ax,ay=(x2-x1)*.4,(y2-y1)*.4  # ellipse covers central 80% of diameter
    xa,xb=max(0,math.floor(cx-ax)),min(w,math.ceil(cx+ax))
    ya,yb=max(0,math.floor(cy-ay)),min(h,math.ceil(cy+ay))
    vals=np.array([]); total=0
    if ax>0 and ay>0 and xb>xa and yb>ya:
        yy,xx=np.mgrid[ya:yb,xa:xb]
        mask=((xx+.5-cx)/ax)**2+((yy+.5-cy)/ay)**2<=1
        samples=depth[ya:yb,xa:xb][mask]; total=int(samples.size)
        vals=samples[np.isfinite(samples)&(samples>0)]
    count=int(vals.size);fraction=count/max(1,total)
    median=float(np.median(vals)) if count else None
    spread=float(np.subtract(*np.percentile(vals,[75,25]))) if count else None
    relative=spread/median if median else None
    status=('ball_too_small' if total<12 else 'insufficient_stereo_depth' if count<12 or fraction<.5
            else 'mixed_depth_region' if relative>.15 else 'valid_depth')
    valid=status=='valid_depth'
    result=dict(depth_m=median if valid else None,valid_depth_count=count,
                depth_sample_count=total,depth_min_valid_count=12,
                valid_depth_fraction=float(fraction),depth_iqr_m=spread,
                depth_relative_iqr=relative,quality_status=status,
                measurement_status='projected_bbox_estimate' if valid else status,
                projected_bbox_width_m=None,projected_bbox_height_m=None,
                depth_sampling_method='ball_inner_ellipse')
    if valid and x1>0 and y1>0 and x2<w and y2<h:
        result.update(projected_bbox_width_m=(x2-x1)*median/geometry.fx,
                      projected_bbox_height_m=(y2-y1)*median/geometry.fy)
    elif valid:result['measurement_status']='partial_box_size_unknown'
    return result


def detail_box_records(result, offset, existing, enabled, depth, frame_id, minimum):
    from detection_filters import BALL_CATEGORIES
    from perception_quality import box_iou
    from depth import box_depth
    rows=[]
    for box,cls,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
        cls=int(cls);name=result.names[cls]
        if cls not in enabled or name not in BALL_CATEGORIES or score<minimum:continue
        box=[box[0]+offset[0],box[1]+offset[1],box[2]+offset[0],box[3]+offset[1]]
        if any(r['object'] in BALL_CATEGORIES and box_iou(box,r['bbox_xyxy'])>.35 for r in existing+rows):continue
        rows.append(dict(frame_id=frame_id,track_id=None,object=name,bbox_xyxy=box,
                         detection_confidence=float(score),detection_method='yolo_detail',
                         depth_m=box_depth(depth,box,inner=.4)))
    return rows


def draw_detail_boxes(image, records, cv2):
    for r in records:
        if r.get('detection_method')!='yolo_detail':continue
        x1,y1,x2,y2=map(int,r['bbox_xyxy'])
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,220,255),2)
        cv2.putText(image,f"{r['object']} {r['detection_confidence']:.2f}",(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1,cv2.LINE_AA)


def detail_region(frame_shape, result):
    """Focus near current ball proposals; fall back to central table area."""
    h,w=frame_shape[:2]
    boxes=[b for b,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist())
           if result.names[int(c)] in ('ball','sports ball','billiard ball','cue ball','pool ball')]
    if not boxes:return int(w*.3),int(h*.45),int(w*.7),int(h*.85)
    x1,y1=min(b[0] for b in boxes),min(b[1] for b in boxes)
    x2,y2=max(b[2] for b in boxes),max(b[3] for b in boxes)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    rw,rh=min(w,max(160,x2-x1+96)),min(h,max(160,y2-y1+96))
    x1=max(0,min(w-rw,cx-rw/2));y1=max(0,min(h-rh,cy-rh/2))
    return int(x1),int(y1),int(x1+rw),int(y1+rh)
