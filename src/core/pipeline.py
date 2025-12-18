import base64
import cv2
import numpy as np

"""
file: pipeline.py

This script contains the service that performs the pipelining of the images by its processing via the services.

Also found here are the functions for b64 encoding/decoding.
"""

class PipelineProcessor:
    @staticmethod
    def B64_to_CV(b64_string):
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)

        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def CV_to_B64(image):
        success, buffer = cv2.imencode(".jpg", image)
        
        if not success:
            raise ValueError("Error: cannot encode cv2 image to b64")
        
        b64_bytes = base64.b64encode(buffer)
        b64_string = b64_bytes.decode("utf-8")

        return f"data:image/jpeg;base64,{b64_string}"

    @staticmethod
    def Process_Frame(b64_string, model):
        #utilize YOLOService singleton for pre-processing and image detection to produce face coordinates
        #utilize VideoService for drawing overlays on the coordinates
        #return image as bytes
        
        cv_img = PipelineProcessor.B64_to_CV(b64_string) 
        results = model._Detect_Image(cv_img)
        annotated_frame = results[0].plot()

        return PipelineProcessor.CV_to_B64(annotated_frame)

