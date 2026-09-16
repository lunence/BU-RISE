from localization.model import Model
from localization.apriltag import AprilTag
from localization.world_frame import WorldFrame

"""
model's predicted poses + homography to return real world coords
"""

class Detector:
    def __init__(self, model_path, init_image = None):
        self.model = Model(model_path)
        self.apriltag = AprilTag()
        self.worldframe = WorldFrame(init_image)

    def predict_pose(self, frame):
        pred_center, pred_orientation = self.model.predict(frame)

        pred_center_world = self.worldframe.pixel_to_world(pred_center)

        return pred_center_world, pred_orientation
    
    def ground_truth_pose(self, frame):
        gt_center, gt_orientation = self.apriltag.get_ground_truth(frame)

        gt_center_world = self.worldframe.pixel_to_world(gt_center)

        return gt_center_world, gt_orientation