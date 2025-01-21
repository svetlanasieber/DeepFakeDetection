def predict_face(image_path, model):
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    if prediction > 0.5:
        return "Fake"
    else:
        return "Real"

print(predict_face("test_face.jpg", model))
