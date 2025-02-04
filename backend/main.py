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

# Load OpenCV's pre-trained face and facial landmark detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_mesh = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def detect_nose_region(image: np.ndarray) -> tuple:
    """
    Detect nose region using OpenCV
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
            
        # Get the largest face
        x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
        
        # Define nose region (approximately center of face)
        nose_x = x + w//4
        nose_y = y + h//3
        nose_w = w//2
        nose_h = h//3
        
        # Create nose points for mask
        nose_points = np.array([
            [nose_x, nose_y],
            [nose_x + nose_w, nose_y],
            [nose_x + nose_w, nose_y + nose_h],
            [nose_x, nose_y + nose_h]
        ])
        
        return (nose_x, nose_y, nose_x + nose_w, nose_y + nose_h), nose_points
        
    except Exception as e:
        logger.error(f"Error in nose detection: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to detect nose region: {str(e)}")

def enhance_nose_region(image: np.ndarray, style: str, nose_region: tuple, nose_points: np.ndarray) -> np.ndarray:
    """
    Enhanced nose region processing with advanced features
    """
    try:
        x1, y1, x2, y2 = nose_region
        enhanced = image.copy()
        nose_area = enhanced[y1:y2, x1:x2]
        
        # Create a mask for smooth blending
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask[y1:y2, x1:x2], nose_points - np.array([x1, y1]), 255)
        mask = cv2.GaussianBlur(mask[y1:y2, x1:x2], (21, 21), 11)
        
        # Style-specific enhancements
        if style == "natural":
            # Subtle enhancement with skin texture preservation
            enhanced_nose = cv2.detailEnhance(nose_area, sigma_s=10, sigma_r=0.15)
            enhanced_nose = cv2.GaussianBlur(enhanced_nose, (3, 3), 0)
            
        elif style == "refined":
            # Enhanced definition with balanced sophistication
            enhanced_nose = cv2.detailEnhance(nose_area, sigma_s=15, sigma_r=0.2)
            enhanced_nose = cv2.bilateralFilter(enhanced_nose, 9, 75, 75)
            
            # Add subtle contours
            edges = cv2.Laplacian(nose_area, cv2.CV_64F).astype(np.uint8)
            enhanced_nose = cv2.addWeighted(enhanced_nose, 0.85, edges, 0.15, 0)
            
        elif style == "elegant":
            # Sophisticated enhancement with subtle contouring
            enhanced_nose = cv2.detailEnhance(nose_area, sigma_s=20, sigma_r=0.25)
            enhanced_nose = cv2.bilateralFilter(enhanced_nose, 11, 90, 90)
            
            # Add highlights and shadows
            light = cv2.addWeighted(enhanced_nose, 1.2, enhanced_nose, 0, 5)
            shadow = cv2.addWeighted(enhanced_nose, 0.8, enhanced_nose, 0, -5)
            enhanced_nose = cv2.addWeighted(light, 0.7, shadow, 0.3, 0)
            
        elif style == "sculpted":
            # Dramatic enhancement with defined contours
            enhanced_nose = cv2.detailEnhance(nose_area, sigma_s=25, sigma_r=0.3)
            
            # Enhance edges and contours
            edges = cv2.Laplacian(nose_area, cv2.CV_64F).astype(np.uint8)
            enhanced_nose = cv2.addWeighted(enhanced_nose, 0.7, edges, 0.3, 0)
            
            # Add depth
            dark = cv2.addWeighted(enhanced_nose, 0.7, enhanced_nose, 0, -10)
            light = cv2.addWeighted(enhanced_nose, 1.3, enhanced_nose, 0, 10)
            enhanced_nose = cv2.addWeighted(dark, 0.5, light, 0.5, 0)
            
        elif style == "conservative":
            # Minimal enhancement for subtle refinement
            enhanced_nose = cv2.detailEnhance(nose_area, sigma_s=5, sigma_r=0.1)
            enhanced_nose = cv2.GaussianBlur(enhanced_nose, (3, 3), 0)
            
        else:  # balanced
            # Harmonious blend of enhancement and naturalness
            enhanced_nose = cv2.detailEnhance(nose_area, sigma_s=12, sigma_r=0.18)
            enhanced_nose = cv2.bilateralFilter(enhanced_nose, 7, 50, 50)
        
        # Advanced color correction
        lab = cv2.cvtColor(enhanced_nose, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_nose = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Apply skin smoothing while preserving texture
        enhanced_nose = cv2.bilateralFilter(enhanced_nose, 9, 75, 75)
        
        # Enhance local contrast
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        enhanced_nose = cv2.morphologyEx(enhanced_nose, cv2.MORPH_CLOSE, kernel)
        
        # Apply high-frequency detail enhancement
        gaussian = cv2.GaussianBlur(enhanced_nose, (0, 0), 3.0)
        unsharp_mask = cv2.addWeighted(enhanced_nose, 2.0, gaussian, -1.0, 0)
        enhanced_nose = cv2.addWeighted(enhanced_nose, 0.7, unsharp_mask, 0.3, 0)
        
        # Blend enhanced nose with original using the mask
        blended = cv2.addWeighted(
            enhanced_nose, 0.7,
            nose_area, 0.3,
            0
        )
        
        # Create a gradient mask for smoother blending
        gradient_mask = cv2.GaussianBlur(mask, (21, 21), 11)
        gradient_mask = gradient_mask / 255.0
        
        # Apply the gradient mask
        enhanced_nose = cv2.multiply(
            enhanced_nose.astype(float),
            cv2.cvtColor(gradient_mask, cv2.COLOR_GRAY2BGR)
        ).astype(np.uint8)
        original_contribution = cv2.multiply(
            nose_area.astype(float),
            cv2.cvtColor(1.0 - gradient_mask, cv2.COLOR_GRAY2BGR)
        ).astype(np.uint8)
        enhanced_nose = cv2.add(enhanced_nose, original_contribution)
        
        # Apply the enhanced region back to the image
        enhanced[y1:y2, x1:x2] = enhanced_nose
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error in nose enhancement: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to enhance nose region: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
