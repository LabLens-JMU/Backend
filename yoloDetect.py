import cv2
import numpy as np
import time
import json
import requests
from picamera2 import Picamera2
from ultralytics import YOLO

# CONFIG, general information
API_URL       = "https://lab-lens.site/api/occupancy/event"
API_KEY       = "lablens-secret"
CAMERA_ID     = "cam-2"
SEND_INTERVAL = 60    # Seconds between posts to api

OCCUPANCY_CLASSES = {0}    # person is identified by 0 in yolov8
MIN_CONFIDENCE    = 0.50        # require at least 50% confidence of human
ZONE_COVERAGE     = 0.5               # Zone most be at least 50% occupied by human


# CAMERA CONFIG, starting the camera
picam = Picamera2()
picam.configure(picam.create_video_configuration(
    main={"format": "RGB888", "size": (1920, 1080)}
))
picam.start()
time.sleep(2)

# YOLO CONFIG
print("Loading YOLOv8 model...")
model = YOLO("yolov8n.pt")
print("Model ready.\n")

# POLYGON ZONE CREATION
print("--- SETUP MODE ---")
print("Click points around each desk zone (as many as you need)")
print("  ENTER = close & save current zone")
print("  r     = redo current zone")
print("  q     = done with all zones\n")

zones       = {}   # name -> np.array of points, shape (N,2)
zone_count  = 1
current_pts = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current_pts.append([x, y])

cv2.namedWindow("Setup Zones")
cv2.setMouseCallback("Setup Zones", click_event)

COLORS = [
    (0, 255, 255), (255, 0, 255), (255, 165, 0),
    (0, 200, 255), (180, 255, 0), (255, 80, 80),
]

while True:
    frame = picam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    temp  = frame.copy()

    # Draw finished zones
    for i, (name, pts) in enumerate(zones.items(), start=31):
        color = COLORS[i % len(COLORS)]
        cv2.polylines(temp, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(temp, name, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw current in-progress polygon
    if current_pts:
        pts_arr = np.array(current_pts, dtype=np.int32)
        for p in current_pts:
            cv2.circle(temp, tuple(p), 4, (0, 255, 255), -1)
        if len(current_pts) > 1:
            cv2.polylines(temp, [pts_arr], False, (0, 255, 255), 2)
        if len(current_pts) > 2:
            cv2.line(temp, tuple(current_pts[-1]), tuple(current_pts[0]),
                     (0, 255, 255), 1)

    cv2.putText(temp,
                f"zone_{zone_count}: {len(current_pts)} pts | ENTER=save  r=redo  q=done",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Setup Zones", temp)
    key = cv2.waitKey(1)

    if key == 13:  # ENTER
        if len(current_pts) >= 3:
            pts_arr = np.array(current_pts, dtype=np.int32)
            zones[f"zone_{zone_count}"] = pts_arr
            print(f"  Saved zone_{zone_count} with {len(current_pts)} points")
            zone_count += 1
            current_pts = []
        else:
            print("  Need at least 3 points!")

    elif key == ord('r'):
        print("  Cleared — start clicking again")
        current_pts = []

    elif key == ord('q'):
        break

cv2.destroyWindow("Setup Zones")
print(f"\n{len(zones)} zones defined.\n")

if not zones:
    print("No zones defined: exiting.")
    picam.stop()
    exit()

# Save zones as coordinates
frame_shape = (1080, 1920)

def make_mask(pts, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask

zone_masks = {name: make_mask(pts, frame_shape) for name, pts in zones.items()}
zone_areas = {name: int(mask.sum()) for name, mask in zone_masks.items()}

# LIVE MONITORING
print("--- LIVE MONITORING ---")
print("Press 'q' to quit\n")

last_send = {}

DWELL_SECONDS = 5    # wait 5s before changing state (change to 120s for final)
GRACE_SECONDS = 15    # hold state for 15s to allow for yolo flicker

candidate_start = {name: None for name in zones}
confirmed_occupied = {name: 0 for name in zones}
best_conf_confirmed = {name: 0.0 for name in zones}
last_seen = {name: 0.0 for name in zones}


while True:
    frame = picam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w  = frame.shape[:2]

    results    = model(frame, verbose=False)
    detections = []
    if results and len(results[0].boxes):
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in OCCUPANCY_CLASSES and conf >= MIN_CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append((x1, y1, x2, y2, conf))

    now = time.time()

    for i, (name, pts) in enumerate(zones.items()):
        color     = COLORS[i % len(COLORS)]
        z_area    = zone_areas[name]
        z_mask    = zone_masks[name]
        raw_occupied  = 0
        best_conf = 0.0

        for (dx1, dy1, dx2, dy2, conf) in detections:
            box_mask = np.zeros((h, w), dtype=np.uint8)
            bx1, by1 = max(0, dx1), max(0, dy1)
            bx2, by2 = min(w, dx2), min(h, dy2)
            box_mask[by1:by2, bx1:bx2] = 1

            overlap = int(np.logical_and(z_mask, box_mask).sum())
            box_area = (bx2 - bx1) * (by2 - by1)
            if box_area > 0 and overlap / z_area >= ZONE_COVERAGE:
                raw_occupied = 1
                if conf > best_conf:
                    best_conf = conf

        
        # Compute waiting confirmation
        if raw_occupied == 1:
            last_seen[name] = now
            if candidate_start[name] is None:
                candidate_start[name] = now

            if now - candidate_start[name] >= DWELL_SECONDS:
                confirmed_occupied[name] = 1
                best_conf_confirmed[name] = best_conf
        else:
            if now - last_seen[name] >= GRACE_SECONDS:
                candidate_start[name] = None
                confirmed_occupied[name] = 0
                best_conf_confirmed[name] = 0.0

        # GUI update
        cv2.polylines(frame, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        status = "Occupied" if confirmed_occupied[name] else "Free"
        label_color = (0, 0, 255) if raw_occupied else (0, 255, 0)
        cv2.putText(frame, f"{name}: {status}",
                    (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        # POST to API
        if now - last_send.get(name, 0) >= SEND_INTERVAL:
            base = 30
            computer_id = base + int(name.split("_")[1])
            #Exact data center receives in JSON:
            payload = {
                "camera_id":   CAMERA_ID,
                "computer_id": computer_id,
                "ts_ms":       int(now * 1000),
                "occupied":    confirmed_occupied[name],
                "confidence":  round(best_conf_confirmed[name], 3),
            }
            print(json.dumps(payload))
            try:
                requests.post(API_URL, json=payload,
                              headers={"Content-Type": "application/json",
                                       "x-api-key": API_KEY},
                              timeout=5)
            except requests.RequestException as e:
                print(f"  POST failed: {e}")
            last_send[name] = now

    # GUI visual cue of detection:
    for (dx1, dy1, dx2, dy2, conf) in detections:
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 165, 255), 1)

    cv2.imshow("Live Occupancy Monitor", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# KILL PROGRAM
picam.stop()
cv2.destroyAllWindows()




