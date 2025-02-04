# New utility functions
import numpy as np
from PIL import Image
import cv2

def detect_face(image):
    """
    Detect face in the image using OpenCV's Haar Cascade
    """
    # Convert PIL Image to cv2 format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(img_cv, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first face
        return image.crop((x, y, x+w, y+h))
    return image

def extract_nose_region(face_image):
    """
    Extract nose region using simple proportions
    """
    width, height = face_image.size
    # Approximate nose position
    nose_width = width // 3
    nose_height = height // 3
    left = (width - nose_width) // 2
    top = height // 3  # Start from 1/3 down
    right = left + nose_width
    bottom = top + nose_height
    
    return face_image.crop((left, top, right, bottom))

def generate_nose_mask(nose_region):
    """
    Generate elliptical mask for the nose region
    """
    width, height = nose_region.size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create elliptical mask
    center = (width // 2, height // 2)
    axes = (width // 3, height // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    return Image.fromarray(mask)

def validate_result(image):
    """
    Validate the generated image result
    """
    try:
        if not isinstance(image, Image.Image):
            return False
        
        width, height = image.size
        if width <= 0 or height <= 0:
            return False
            
        if image.getbbox() is None:
            return False
            
        return True
        
    except Exception:
        return False 