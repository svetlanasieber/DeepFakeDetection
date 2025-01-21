import cv2
from mtcnn import MTCNN
from PIL import Image

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{frame_count}.jpg", frame)
        frame_count += 1
    cap.release()

def extract_faces(frame_path, output_folder):
    detector = MTCNN()
    image = Image.open(frame_path)
    image = image.convert('RGB')
    pixels = cv2.imread(frame_path)
    results = detector.detect_faces(pixels)

    for i, result in enumerate(results):
        x, y, width, height = result['box']
        face = pixels[y:y+height, x:x+width]
        face_image = Image.fromarray(face)
        face_image.save(f"{output_folder}/face_{i}.jpg")
