"""Billiards-only categories and plain boxes with numeric model scores."""
BILLIARDS_CATEGORIES=frozenset(('billiard ball','cue ball'))


def billiards_names(names):
    return {k:v for k,v in names.items() if v in BILLIARDS_CATEGORIES}


def draw_score_boxes(image, records, cv2):
    for r in records:
        score=r.get('detection_confidence')
        if score is None:continue
        x1,y1,x2,y2=map(int,r['bbox_xyxy'])
        color=(0,220,255)
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        cv2.putText(image,f'{score*100:.0f}%',(max(0,x1),max(16,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
