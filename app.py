import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from flask import Flask, render_template, Response, jsonify
from tf_keras.applications.mobilenet_v2 import preprocess_input
from tf_keras.preprocessing.image import img_to_array
from tf_keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import threading
import time

app = Flask(__name__)

# Load models
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(r"model\mask_detector.h5")

# Global variables for video stream
vs = None
lock = threading.Lock()
current_stats = {"masks": 0, "no_masks": 0, "confidence": "0.00"}

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def generate_frames():
    global vs, lock, current_stats
    # Ensure stream is started if someone directly hits the video URL
    if vs is None:
        vs = VideoStream(src=0).start()
        time.sleep(2.0) # wait for camera to warm up
    
    while True:
        with lock:
            if vs is None:
                break
            frame = vs.read()
        
        if frame is None:
            continue
            
        frame = imutils.resize(frame, width=800)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        frame_masks = 0
        frame_no_masks = 0
        highest_conf = 0.0

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            conf = max(mask, withoutMask) * 100
            if conf > highest_conf:
                highest_conf = conf

            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
                frame_masks += 1
            else:
                label = "No Mask"
                color = (0, 0, 255)
                frame_no_masks += 1
                
            label = "{}: {:.2f}%".format(label, conf)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Update global stats
        current_stats["masks"] = frame_masks
        current_stats["no_masks"] = frame_no_masks
        current_stats["confidence"] = "{:.2f}".format(highest_conf)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_stream", methods=["POST"])
def start_stream():
    global vs
    with lock:
        if vs is None:
            vs = VideoStream(src=0).start()
            time.sleep(2.0)
    return jsonify({"status": "started"})

@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    global vs
    with lock:
        if vs is not None:
            vs.stop()
            vs = None
    return jsonify({"status": "stopped"})

@app.route("/stats")
def get_stats():
    global vs, current_stats
    if vs is None:
        return jsonify({"masks": 0, "no_masks": 0, "confidence": "0.00"})
    return jsonify(current_stats)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)
