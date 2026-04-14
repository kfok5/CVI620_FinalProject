import os
print('🚀 Starting Inference Server...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import socketio
import eventlet
import numpy as np 
from keras.models import load_model
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
app = Flask(__name__)

maxSpeed = 10

def preprocessing(img):
    h, w, _ = img.shape
    img = img[int(h * 0.4):int(h * 0.85), :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)   
    img = cv2.resize(img, (200, 66))
    img = img / 255

    return img

@sio.on('telemetry')
def telemetry(sid, data):
    try:
        speed = float(data['speed'])
        
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = preprocessing(image)
        image = np.array([image])

        steering = float(model.predict(image, verbose=0)[0])
        steering *= 1.5
        throttle = 1.0 - speed / maxSpeed
        print(f"Steering: {steering:.4f}, Throttle: {throttle:.4f}, Speed: {speed}")

        sendControl(steering, throttle)

    except Exception as e:
        print(e)

@sio.on('connect')
def connect(sid, environ):
    sendControl(0, 0)

def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    })

if __name__ == "__main__":
    model = load_model('self_driving_model.h5', compile=False)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)