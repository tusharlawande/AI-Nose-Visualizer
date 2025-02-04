import urllib.request
import os

def download_yunet_model():
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    model_path = "face_detection_yunet_2023mar.onnx"
    
    if not os.path.exists(model_path):
        print(f"Downloading YuNet model from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete!")
    else:
        print("Model file already exists.")

if __name__ == "__main__":
    download_yunet_model()
