import cv2
import base64
import asyncio
import json
import time
from typing import Dict
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware

# ---------------- model ----------------
model = YOLO("yolov8n.pt")


detected_objects_store: Dict[str, Dict[int, dict]] = {}


# ---------------- app ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TRANSMIT_PASSWORD = "tx_secret"
RECEIVE_PASSWORD = "rx_secret"


latest_payload = {}
receivers = set()

# ---------------- persistence ----------------
LOG_FILE = "detections_log.jsonl"

# ---------------- camera registry ----------------
cameras: Dict[str, cv2.VideoCapture] = {}

# ---------------- object memory ----------------
OBJECT_TTL = 5.0
IOU_THRESHOLD = 0.35

object_memories: Dict[str, Dict[int, dict]] = {}
object_id_seq: Dict[str, int] = {}

# ---------------- utils ----------------
def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode()


def build_metadata(cls, bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    fh, fw = frame_shape[:2]

    scale_w = w / fw
    scale_h = h / fh

    meta = {
        "width_px": round(w, 2),
        "height_px": round(h, 2),
        "scale_w": round(scale_w, 4),
        "scale_h": round(scale_h, 4)
    }

    if cls == "person":
        meta["estimated_age_group"] = "unknown"
        meta["estimated_height_ratio"] = round(scale_h, 4)

    if cls in ["car", "truck", "bus", "bicycle"]:
        meta["object_type"] = "vehicle"

    if cls in ["building"]:
        meta["structure"] = True

    return meta

def log_metadata(entry):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2,(y1+y2)/2)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    return inter / (area_a + area_b - inter)

def match_persistent(camera_key, detection):
    store = detected_objects_store.get(camera_key)
    if not store:
        return None

    db = detection["bbox"]
    dc = center(db)

    best = None
    best_score = 0

    for oid, obj in store.items():
        if obj["class"] != detection["cls"]:
            continue

        pb = obj["bbox"]

        score = iou(pb, db)

        pc = center(pb)
        dist = ((dc[0]-pc[0])**2 + (dc[1]-pc[1])**2)**0.5

        if score > 0.4 or dist < 80:
            if score > best_score:
                best = oid
                best_score = score

    return best

def register_objects(camera_key, detections, ts):
    if camera_key not in object_memories:
        object_memories[camera_key] = {}
        object_id_seq[camera_key] = 0

    memory = object_memories[camera_key]

    for oid in list(memory.keys()):
        if ts - memory[oid]["last_seen"] > OBJECT_TTL:
            del memory[oid]

    newly_found = []

    for d in detections:
        matched = False
        for oid, o in memory.items():
            if o["cls"] == d["cls"] and iou(o["bbox"], d["bbox"]) > IOU_THRESHOLD:
                o["bbox"] = d["bbox"]
                o["last_seen"] = ts
                matched = True
                break

        if not matched:
            oid = object_id_seq[camera_key]
            object_id_seq[camera_key] += 1
            memory[oid] = {
                "id": oid,
                "cls": d["cls"],
                "bbox": d["bbox"],
                "first_seen": ts,
                "last_seen": ts,
            }
            newly_found.append(memory[oid])

    return newly_found, list(memory.values())

def persist_objects(camera_key, newly_found, frame):

    if camera_key not in detected_objects_store:
        detected_objects_store[camera_key] = {}

    store = detected_objects_store[camera_key]

    for obj in newly_found:

        # attempt match with already stored objects
        existing = match_persistent(camera_key, {
            "cls": obj["cls"],
            "bbox": obj["bbox"]
        })

        if existing is not None:
            store[existing]["bbox"] = obj["bbox"]
            continue

        oid = obj["id"]

        meta = build_metadata(
            obj["cls"],
            obj["bbox"],
            frame.shape
        )

        store[oid] = {
            "id": oid,
            "class": obj["cls"],
            "first_seen": obj["first_seen"],
            "bbox": obj["bbox"],
            "metadata": meta
        }

def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = f'{d["cls"]} {d["conf"]:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-h-6), (x1+w+4, y1), (0,255,0), -1)
        cv2.putText(
            frame,
            label,
            (x1+2, y1-4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            1,
            cv2.LINE_AA
        )

# ---------------- visual modes ----------------
def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

def high_contrast(frame):
    return cv2.convertScaleAbs(frame, alpha=1.5, beta=20)

def edge_ir_like(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 60, 120)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)

def thermal_simulated(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)

STREAMS = {
    "rgb": lambda f: f,
    "low_light": enhance_low_light,
    "contrast": high_contrast,
    "edge": edge_ir_like,
    "thermal": thermal_simulated,
}

# ---------------- websocket ----------------
@app.websocket("/ws")
async def stream(
    websocket: WebSocket,
    camera: str = Query("0")
):
    await websocket.accept()
    print("WS connected:", camera)

    if camera not in cameras:
        cam_index = int(camera)
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {cam_index}")

        cameras[camera] = cap

    cap = cameras[camera]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            ts = time.time()

            res = model(frame, conf=0.4)[0]

            detections = [
                {
                    "cls": model.names[int(box.cls)],
                    "conf": round(float(box.conf), 3),
                    "bbox": list(map(float, box.xyxy[0])),
                }
                for box in res.boxes
            ]

            newly_found, all_objects = register_objects(camera, detections, ts)
            persist_objects(camera, newly_found, frame)
            
            annotated = frame.copy()
            draw_detections(annotated, detections)

            if newly_found:
                log_metadata({
                    "timestamp": ts,
                    "camera": camera,
                    "new_objects": newly_found
                })

            payload = {
                "timestamp": ts,
                "streams": {
                    name: encode_frame(fn(annotated))
                    for name, fn in STREAMS.items()
                },
                "metadata": {
                    "realtime": detections,
                    "objects_found": list(detected_objects_store.get(camera, {}).values())
                }
            }

            await websocket.send_json(payload)
            await asyncio.sleep(0.03)

    except Exception as e:
        print("WS ERROR:", e)

    finally:
        print("WS closed:", camera)


@app.websocket("/ws/transmit")
async def transmit(websocket: WebSocket, password: str = Query(...)):
    if password != TRANSMIT_PASSWORD:
        await websocket.close()
        return

    await websocket.accept()

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.03)
                continue

            ts = time.time()

            res = model(frame, conf=0.4)[0]

            detections = [
                {
                    "cls": model.names[int(box.cls)],
                    "conf": round(float(box.conf), 3),
                    "bbox": list(map(float, box.xyxy[0]))
                }
                for box in res.boxes
            ]

            newly_found, _ = register_objects("0", detections, ts)
            persist_objects("0", newly_found, frame)

            annotated = frame.copy()
            draw_detections(annotated, detections)

            payload = {
                "timestamp": ts,
                "streams": {
                    name: encode_frame(fn(annotated))
                    for name, fn in STREAMS.items()
                },
                "metadata": {
                    "realtime": detections,
                    "objects_found": list(detected_objects_store.get("0", {}).values())
                }
            }

            dead = []

            for r in receivers:
                try:
                    await r.send_json(payload)
                except:
                    dead.append(r)

            for d in dead:
                receivers.remove(d)

            await asyncio.sleep(0.03)

    finally:
        cap.release()



@app.websocket("/ws/receive")
async def receive(
    websocket: WebSocket,
    password: str = Query(...)
):
    if password != RECEIVE_PASSWORD:
        await websocket.close()
        return

    await websocket.accept()
    receivers.add(websocket)

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        receivers.remove(websocket)


