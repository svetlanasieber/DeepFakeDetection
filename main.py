import os
from scripts.preprocess import extract_frames, extract_faces
from scripts.train import train_model
from scripts.predict import predict_face

def main():
  
    extract_frames("data/raw/deepfake_video.mp4", "data/frames")

  
    frame_path = "data/frames/frame_0.jpg"
    extract_faces(frame_path, "data/faces")

  
    train_model("data/faces", "models/deepfake_model.h5")

 
    result = predict_face("data/faces/face_0.jpg", "models/deepfake_model.h5")
    print(f"Prediction: {result}")

if __name__ == "__main__":
    main()
