import cv2
from localization.camera import Camera
from localization.model import Model

weights = "localization/saved_models/best_finetuning_model_lr1e-3.pth"

model = Model(weights)

frame = cv2.imread("test_scripts/limo1.jpg")

center, heading = model.predict(frame)

print(center)
print(heading)