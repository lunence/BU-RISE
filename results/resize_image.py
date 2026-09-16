import cv2

def crop_to_ratio(image):
    height, width = image.shape[:2]
    crop_width = height * 4 // 3
    x_start = (width - crop_width) // 2   # 160
    x_end = x_start + crop_width          # 1120

    cropped = image[:, x_start:x_end]
    return cropped

image = cv2.imread("/Users/mooncyli/Desktop/BU_RISE/BU-RISE/results/video_dataset_ex.png")

cropped = crop_to_ratio(image)

resized = cv2.resize(cropped, (160, 120), interpolation=cv2.INTER_CUBIC)

cv2.imwrite("/Users/mooncyli/Desktop/BU_RISE/BU-RISE/results/resized.png", resized)
