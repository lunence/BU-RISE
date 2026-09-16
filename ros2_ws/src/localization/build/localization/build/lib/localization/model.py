import torch

from . import config
from localization.robot_detector_model.model import GridNet
from localization.camera import Camera

class Model:
    def __init__(self, model_path):
        #load trained model
        self.device = torch.device('cpu')

        self.model = GridNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, frame):
        # resize and reformat for model
        frame = Camera.model_prep(frame, self.device)

        #run inference
        with torch.no_grad():
            logits = self.model(frame)

        # center pred (normalized)
        pred_center = logits["center"]

        # orientation pred (degrees)
        pred_orientation = logits["orientation"].argmax(dim=1) * 5

        return pred_center, pred_orientation

    # TODO: implement all of these functions (paste from detect.py)
    def calculate_errors(self):
        return

    def print_predictions(self):
        return

    def plot_predictions(self):
        return