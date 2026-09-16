import torch

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

        return logits

    