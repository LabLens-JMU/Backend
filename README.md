# LabLens – Backend (Hardware / Edge Layer)

> Real-time computer lab occupancy detection using machine vision.

The Backend is the physical edge layer of the LabLens system. It uses a Raspberry Pi 5 and camera module to capture live video, runs occupancy detection using YOLOv8 and OpenCV, and transmits structured occupancy data to the Center API in real time.

---

## How It Works

1. The Raspberry Pi camera continuously captures live frames from the lab environment.
2. OpenCV prepares each frame and manages predefined desk zone regions.
3. YOLOv8 runs inference on each frame to detect people and generate bounding boxes.
4. Detected objects are mapped to desk zones (Regions of Interest) to determine seat status.
5. Time-based logic distinguishes between **Occupied**, **Idle/Away**, and **Available** states.
6. Results are structured into JSON and sent to the Center via a POST request.

---

## Seat States

| State | Meaning |
|-------|---------|
| 🔴 Occupied | A person is actively at the workstation |
| 🟡 Idle / Away | Seat is in use but the person is temporarily away |
| 🟢 Available | Seat is unoccupied |

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Raspberry Pi 5 | Primary edge processing unit |
| Raspberry Pi Camera Module 2 | Live video capture |
| YOLOv8 | Real-time human presence detection |
| OpenCV | Frame capture, processing, and zone definition |
| Python | Detection logic and API communication |
| HTTP / JSON | Structured data transmission to the Center |

---

## Detection Logic

- **Bounding Boxes** — YOLOv8 localizes detected individuals within the camera frame.
- **Regions of Interest (ROI)** — Each desk is mapped to a polygon zone drawn during setup.
- **Confidence Threshold** — Minimum 50% confidence required (`MIN_CONFIDENCE = 0.50`).
- **Zone Coverage** — A detection must overlap at least 50% of a zone to count (`ZONE_COVERAGE = 0.5`).
- **Dwell Time** — A person must be present for 5 seconds before a seat is marked Occupied (`DWELL_SECONDS = 5`).
- **Grace Period** — A seat stays Occupied for 15 seconds after a person leaves to handle detection flicker (`GRACE_SECONDS = 15`).

---

## API Communication

Occupancy events are sent to the Center every 60 seconds per zone.

**Endpoint:** `POST /api/occupancy/event`

**Headers:**
```
Content-Type: application/json
x-api-key: lablens-secret
```

**Payload:**
```json
{
  "camera_id": "cam-2",
  "computer_id": 31,
  "ts_ms": 1773502005000,
  "occupied": 1,
  "confidence": 0.95
}
```

---

## How to Run

### Prerequisites
- Raspberry Pi 5 with camera module enabled
- Python 3 installed
- Required packages: `ultralytics`, `opencv-python`, `picamera2`, `requests`

### Setup

```bash
# Install dependencies
pip install ultralytics opencv-python picamera2 requests

# Run the detection script
python3 detection.py
```

On first run, the script enters **Setup Mode** — use your mouse to click polygon zones around each desk in the camera frame, then press:
- `ENTER` — save the current zone
- `R` — redo the current zone
- `Q` — finish setup and begin live monitoring

---

## Remote Access

The backend can be accessed and managed remotely using **Raspberry Pi Connect**.

---

## Known Limitations

- Detection accuracy can be affected by lighting conditions, occlusion, and camera angle.
- No heartbeat monitoring — the system cannot detect if the camera goes offline.
- Real-time computer vision on edge hardware introduces performance constraints.

---

## Future Improvements

- Model tuning and hardware acceleration for better performance
- Support for multiple cameras and larger lab environments
- Improved 3D printed enclosure for cleaner deployment
- Heartbeat / last-seen monitoring per device

---

## Related Repositories

- [LabLens Center (Backend API)](https://github.com/LabLens-JMU)
- [LabLens Frontend](https://github.com/LabLens-JMU/Frontend)
- [Live Dashboard](https://lab-lens.site)
