import cv2
import math
from pupil_apriltags import Detector

from . import config
from localization.camera import Camera
from localization.utils import normalize_coords

class AprilTag:
    detector = Detector(families='tag36h11',
                    nthreads=1,
                    quad_decimate=1.0,
                    quad_sigma=0.8,
                    refine_edges=1,
                    decode_sharpening=0.5,
                    debug=0)

    @staticmethod
    def get_ground_truth(frame):
        # frame = Camera.apriltag_resize(frame)
        tags = AprilTag.get_apriltags(frame, config.TAG_SIZE_LIMO)
        
        if(len(tags)>0):
            cx, cy = tags[0].center.astype(int)
            pose = normalize_coords((cx, cy), config.APRILTAG_WIDTH, config.APRILTAG_HEIGHT, 1, 1)
            rotation_matrix = tags[0].pose_R
            orientation = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0]) * 180 / math.pi
            ground_truth = {
                "center": pose,
                "orientation": (orientation+360)%360,
                "class": 1
            }
        else:
            ground_truth = {
                "center": (0,0),
                "orientation": 0,
                "class": 0
            }
            
        return ground_truth

    @staticmethod
    def get_apriltags(frame, tag_size = config.TAG_SIZE):
        frame = Camera.apriltag_resize(frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downscaled = Camera.apriltag_resize(image)
        # print(downscaled.shape)
    
        # TODO: change back to downscaled
        tags = AprilTag.detector.detect(image, True, config.CAMERA_PARAMS, tag_size)
    
        return tags