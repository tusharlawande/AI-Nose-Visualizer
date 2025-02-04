from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io
import logging
from typing import Optional

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenCV's pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def detect_nose_region(image: np.ndarray):
    """
    Detect nose region using OpenCV cascade classifiers
    """
    try:
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
            
        # Get the largest face
        max_area = 0
        max_face = None
        for face in faces:
            area = face[2] * face[3]
            if area > max_area:
                max_area = area
                max_face = face
                
        x, y, w, h = max_face
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect nose within the face region
        noses = nose_cascade.detectMultiScale(face_roi, 1.3, 5)
        
        if len(noses) == 0:
            raise ValueError("No nose detected in the image")
            
        # Get the largest nose
        nx, ny, nw, nh = noses[0]
        for nose in noses:
            area = nose[2] * nose[3]
            if area > nw * nh:
                nx, ny, nw, nh = nose
                
        # Convert nose coordinates to full image coordinates
        nose_region = (x + nx, y + ny, nw, nh)
        
        # Create nose points (simplified version)
        nose_points = np.array([
            [x + nx + nw//2, y + ny],  # top
            [x + nx, y + ny + nh//2],  # left
            [x + nx + nw, y + ny + nh//2],  # right
            [x + nx + nw//2, y + ny + nh]  # bottom
        ])
        
        return nose_region, nose_points
        
    except Exception as e:
        logger.error(f"Error in nose detection: {str(e)}")
        raise

def enhance_nose_region(image: np.ndarray, style: str, nose_region: tuple, nose_points: np.ndarray):
    """
    Enhanced nose region processing with different styles
    """
    try:
        # Create a copy of the image
        result = image.copy()
        
        # Extract nose region coordinates
        x, y, w, h = nose_region
        
        # Convert to PIL Image for enhancement
        nose_img = Image.fromarray(cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
        
        # Apply style-specific enhancements
        if style == "natural":
            # Subtle enhancement
            enhancer = ImageEnhance.Contrast(nose_img)
            nose_img = enhancer.enhance(1.1)
            enhancer = ImageEnhance.Sharpness(nose_img)
            nose_img = enhancer.enhance(1.1)
            
        elif style == "refined":
            # More pronounced enhancement
            enhancer = ImageEnhance.Contrast(nose_img)
            nose_img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Sharpness(nose_img)
            nose_img = enhancer.enhance(1.3)
            
        elif style == "elegant":
            # Sophisticated enhancement
            nose_img = nose_img.filter(ImageFilter.SMOOTH)
            enhancer = ImageEnhance.Contrast(nose_img)
            nose_img = enhancer.enhance(1.15)
            
        elif style == "sculpted":
            # Strong enhancement
            enhancer = ImageEnhance.Contrast(nose_img)
            nose_img = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Sharpness(nose_img)
            nose_img = enhancer.enhance(1.4)
            
        else:  # conservative or default
            # Minimal enhancement
            enhancer = ImageEnhance.Contrast(nose_img)
            nose_img = enhancer.enhance(1.05)
            
        # Convert back to OpenCV format
        enhanced_nose = cv2.cvtColor(np.array(nose_img), cv2.COLOR_RGB2BGR)
        
        # Blend the enhanced nose region back into the original image
        result[y:y+h, x:x+w] = enhanced_nose
        
        return result
        
    except Exception as e:
        logger.error(f"Error in nose enhancement: {str(e)}")
        raise

@app.post("/process-image/")
async def process_image(
    file: UploadFile = File(...),
    style: str = "natural",
    view_type: str = "frontView"
):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert image to 8-bit unsigned integer format
        image = cv2.convertScaleAbs(image)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Process image based on view type
        if view_type == "frontView":
            nose_region, nose_points = detect_nose_region(image)
            enhanced_image = enhance_nose_region(image, style, nose_region, nose_points)
        else:
            # Add specific processing for side and three-quarter views
            raise HTTPException(status_code=400, detail="Only front view is currently supported")
            
        # Convert back to bytes
        is_success, buffer = cv2.imencode(".png", enhanced_image)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image")
            
        # Return the processed image
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png"
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
