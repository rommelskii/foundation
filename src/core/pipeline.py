"""
file: pipeline.py

This script contains the service that performs the pipelining of the images by its processing via the services
""

class PipelineProcessor:
    @staticmethod
    def Process_Frame(bytes):
        #utilize YOLOService singleton for pre-processing and image detection to produce face coordinates
        #utilize VideoService for drawing overlays on the coordinates
        #return image as bytes
