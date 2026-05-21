import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tkinter import Tk, filedialog
import onnxruntime as ort
from gpiozero import PWMOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep
factory = LGPIOFactory(chip=4)
buzzer = PWMOutputDevice(18, pin_factory=factory, frequency=1000)
root = Tk()
root.withdraw()
filepath = filedialog.askopenfilename()
if not filepath:
    exit()
path = filepath
OUTPUT_VIDEO = 'annotated_output12.mp4'
frame_count = 0
DETECT_MODEL = 'yolo26n.onnx'
CLASSIFY_MODEL = '26class.onnx'
shur = 0.90
CONFIDENCE_CAT = 0.5
SKIP_FRAMES = 2
detector = YOLO(DETECT_MODEL, verbose=False)
classify_session = ort.InferenceSession(CLASSIFY_MODEL, providers=['CPUExecutionProvider'])
input_name = classify_session.get_inputs()[0].name
imgsz = classify_session.get_inputs()[0].shape[2]
class_names = ['normal', 'scratching']
font = ImageFont.load_default()
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise ValueError(f"Cannot open video: {path}")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
alert_active = False
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = 320 / max(h, w)  # Уменьшено: 320 вместо 640
        new_h, new_w = int(h * scale), int(w * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h))
        frame_count += 1
        if frame_count % SKIP_FRAMES == 0:
            out.write(frame)
            continue
        annotated_frame = frame.copy()
        results = detector.predict(rgb_frame, classes=[15], conf=CONFIDENCE_CAT, verbose=False)
        scratching_detected = False
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                cat_crop = rgb_frame[y1:y2, x1:x2]
                if cat_crop.size == 0 or cat_crop.shape[0] < 32 or cat_crop.shape[1] < 32:
                    continue
                resized = cv2.resize(cat_crop, (imgsz, imgsz))
                input_tensor = resized.transpose(2, 0, 1)
                input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0
                outputs = classify_session.run(None, {input_name: input_tensor})
                probs = outputs[0][0]
                top_class_idx = int(np.argmax(probs))
                confidence = float(probs[top_class_idx])
                class_name = class_names[top_class_idx]
                if class_name == 'scratching' and confidence > shur:
                    status_text = f"{class_name}({confidence:.2f})"
                    color = (0, 0, 255)
                    scratching_detected = True
                else:
                    status_text = f"{class_name.capitalize()} ({confidence:.2f})"
                    color = (0, 255, 0)
                pil_image = Image.fromarray(annotated_frame)
                draw = ImageDraw.Draw(pil_image)
                draw.rectangle([x1, y1, x2, y2], outline=tuple(c // 2 for c in color), width=3)
                draw.text((x1, y1 - 30), status_text, fill=color[::-1], font=font)
                annotated_frame = np.array(pil_image)
        if scratching_detected and not alert_active:
            print(f"SCRATCHING! Confidence: {confidence:.3f}")
            #buzzer.value = 0.5
            alert_active = True
        elif not scratching_detected and alert_active:
            #buzzer.value = 0.0
            alert_active = False
        out.write(annotated_frame)
finally:
    cap.release()
    out.release()
    #buzzer.value = 0.0
    #buzzer.close()