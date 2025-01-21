def detect_in_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    detector = MTCNN()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
            face_crop = frame[y:y+height, x:x+width]
            
         
            face_img = cv2.resize(face_crop, (128, 128))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            
            prediction = model.predict(face_img)
            label = "Fake" if prediction > 0.5 else "Real"
            color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
            
            cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_in_video("test_video.mp4", model)
