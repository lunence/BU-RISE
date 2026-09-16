def crop_to_ratio(image):
    height, width = image.shape[:2]
    crop_width = width * 3 // 4
    x_start = (width - crop_width) // 2   # 160
    x_end = x_start + crop_width          # 1120

    cropped = image[:, x_start:x_end]
    return cropped