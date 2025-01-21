from mtcnn import MTCNN
from PIL import Image

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

extract_faces("output_frames/frame_0.jpg", "faces_output")
