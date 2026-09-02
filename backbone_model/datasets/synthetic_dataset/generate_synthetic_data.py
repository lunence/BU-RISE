import cv2
import numpy as np
import random
import json
import os
from utils import normalize_coords
from config import WIDTH, HEIGHT

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds Gaussian noise to an image.
    mean: Average change in pixel value (usually 0).
    sigma: Standard deviation (higher means more noise).
    """
    # Generate random Gaussian noise with the same shape as the image
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    
    # Add the noise to the original image
    noisy_image = image.astype(np.float32) + noise
    
    # Clip the pixel values to keep them in the valid [0, 255] range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def add_salt_and_pepper_noise(image, amount=0.04):
    """
    Adds Salt and Pepper noise to an image.
    amount: Percentage of pixels to affect (e.g., 0.04 = 4%).
    """
    noisy_image = image.copy()
    
    # Add Salt (White pixels)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Add Pepper (Black pixels)
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

def apply_random_augmentation(image):
    choice = random.choice(['gaussian', 'salt_pepper', 'both', 'clean'])
    
    if choice == 'gaussian':
        return add_gaussian_noise(image, sigma=random.randint(10, 40))
    elif choice == 'salt_pepper':
        return add_salt_and_pepper_noise(image, amount=random.uniform(0.01, 0.05))
    elif choice == 'both':
        # Apply Gaussian first, then Salt & Pepper on top
        img_gau = add_gaussian_noise(image, sigma=random.randint(10, 25))
        return add_salt_and_pepper_noise(img_gau, amount=random.uniform(0.01, 0.03))
    else:
        return image # Keep it clean

def robot_present():
    return random.choice([True, False])

def random_background(sequence_folder):
    image_files = []
    
    for file_name in os.listdir(sequence_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(sequence_folder, file_name))
    if not image_files:
        print(f"Could not load image {image_files}")
        return []
    
    image = cv2.imread(image_files[random.randint(0, len(image_files)-1)])

    return image

def generate_synthetic_dataset(size):
    ground_truth = []

    robot_path = "backbone_model/real_world_dataset/real_limo2.png"
    background_path = 'backbone_model/real_world_dataset/backgrounds'

    # 1. Gather all images and extract their physical creation timestamps
    for i in range(size):
        color = (0,0,0)
        # get random background image
        background = random_background(background_path)
        downscale_bg = cv2.resize(background, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        if(robot_present()):

            image = cv2.imread(robot_path)

            if image is None:
                raise FileNotFoundError(f"Could not load image: {robot_path}")

            # Get image dimensions
            height, width = image.shape[:2]
            final_width = int(height/3*4)
            pad_hori = (final_width-width) // 2
            width = final_width
            image = cv2.copyMakeBorder(
                image, 0, 0, pad_hori, pad_hori, 
                borderType=cv2.BORDER_CONSTANT, value=color
            )

            # --- Step 1: Rotation ---
            # Define center, angle, and scale
            center = (width // 2, height // 2)
            angle = random.randint(0, 359)  # degrees
            scale = random.uniform(0.20, 0.30)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            
            # Apply rotation
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=color)

            # resize to height x width
            downscaled = cv2.resize(rotated_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            robot_width = width*scale/final_width*WIDTH 
            robot_height = height*scale/height*HEIGHT
                    
            # --- Step 2: Translation ---
            # Define translation offsets
            # half of frame size - half of robot size
            tx, ty = random.randint(-(WIDTH//2-robot_width//2),(WIDTH//2-robot_width//2)), random.randint(-(HEIGHT//2-robot_height//2),(HEIGHT//2-robot_height//2))  # Right and down
            
            # Create translation matrix
            translation_matrix = np.float32([
                [1, 0, tx],
                [0, 1, ty]
            ])
                    
            # Apply translation on rotated image
            translated_image = cv2.warpAffine(downscaled, translation_matrix, (WIDTH, HEIGHT), borderMode=cv2.BORDER_CONSTANT, borderValue=color)

            # 4. Create the mask from the rotated image
            gray_rotated = cv2.cvtColor(translated_image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_rotated, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # 5. Apply the masks directly to the full images (No offsets or slicing needed!)
            # Black out the foreground area on the background
            background_bg = cv2.bitwise_and(downscale_bg, downscale_bg, mask=mask_inv)

            # Isolate the rotated foreground object (removing its black borders)
            foreground_fg = cv2.bitwise_and(translated_image, translated_image, mask=mask)

            # 6. Combine them together into the final image
            final_result = cv2.add(background_bg, foreground_fg)

            ground_truth.append({
                "center": normalize_coords((tx, ty), WIDTH, HEIGHT, 1, 1, True),
                "orientation": angle,
                "class": 1
            })
        else:
            ground_truth.append({
                "center": (0,0),
                "orientation": 0, 
                "class": 0
            })
            final_result = downscale_bg

        noisy_image = apply_random_augmentation(final_result)
        
        # Save and show result
        cv2.imwrite(f'backbone_model/real_world_dataset/images/transformed_limo_{i}.jpg', noisy_image)

    with open("backbone_model/real_world_dataset/ground_truth.json", "w") as file:
        json.dump(ground_truth, file, indent = 4)