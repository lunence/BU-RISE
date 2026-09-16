import cv2
import math
from pupil_apriltags import Detector

from . import config
from localization.camera import Camera

class AprilTag:
    detector = Detector(families='tag36h11',
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

    @staticmethod
    def get_ground_truth(frame):
        tags = AprilTag.get_apriltag_by_image(frame, config.TAG_SIZE_LIMO)
        ground_truth = []
        
        if(len(tags)>0):
            cx, cy = tags[0].center.astype(int)
            rotation_matrix = tags[0].pose_R
            orientation = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / math.pi
            ground_truth.append({
                "center": (cx, cy),
                "orientation": orientation,
                "class": 1
            })
        return ground_truth

    @staticmethod
    def get_apriltags(frame, tag_size = config.TAG_SIZE):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        downscaled = Camera.apriltag_resize(image)
    
        tags = AprilTag.detector.detect(downscaled, True, config.CAMERA_PARAMS, tag_size)
    
        return tags