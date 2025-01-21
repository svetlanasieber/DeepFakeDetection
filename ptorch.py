import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


model = models.resnet50(pretrained=True)
model.eval() 


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = 'example.jpg' 
img = Image.open(img_path)
img_tensor = preprocess(img).unsqueeze(0) 


with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    print(f"Predicted class index: {predicted.item()}")
