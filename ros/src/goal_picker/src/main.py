#!/usr/bin/env python
from flask import Flask, request
from flask import render_template
from os import path
import numpy as np
from service import Service
import cv2

# set the project root directory as the static folder, you can set others.
app = Flask(__name__)
service = Service()

@app.route('/', methods = ['GET', 'POST'])
def root():
    message = None
    if request.method == 'POST':
        f = request.files['goal_image']
        nparr = np.fromstring(f.read(), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        message = 'file uploaded successfully'
        try:
            service.send_goal_image(img_np)
        except Exception as e:
            message = "call failed with: %s" % e
    return render_template("index.html", message = message)

if __name__ == "__main__":
    service.initialize()
    app.run(debug=True)
    print("Goal picker app started")