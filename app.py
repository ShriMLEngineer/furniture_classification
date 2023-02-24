from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import git
import shutil

app = FastAPI()

repo_url="https://github.com/ShriMLEngineer/furniture_classification.git"
repo_dir = "furniture_classification"

git.Repo.clone_from(repo_url, repo_dir)

# load trained DETR model
path = "furniture_classification/detr-main"

model = torch.hub.load(path, 'detr_resnet50', source="local", pretrained=True, num_classes=4)
checkpoint = torch.load("furniture_classification//outputs/checkpoint.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval();

# Define the classes
CLASSES = [
    'N/A', 'Bed', 'Chair', 'Sofa'
]

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the function for object detection
def detect_objects(img):
    img_t = transform(img).unsqueeze(0)
    output = model(img_t)

    # Get the class and bounding box predictions
    scores = output['pred_logits'].softmax(dim=-1)[0, :, :-1]
    boxes = output['pred_boxes'][0]

    # Apply a threshold to the scores
    threshold = 0.5
    keep = scores.max(-1).values > threshold

    # Filter the boxes and scores
    boxes = boxes[keep]
    scores = scores[keep]

    # Get the predicted classes
    classes = scores.argmax(-1)

    # Convert the classes to class names
    class_names = [CLASSES[c] for c in classes]

    # Return the class names and bounding boxes
    return class_names, boxes.tolist()

# Define the API endpoint for object detection
@app.post('/detect', response_model=dict)
async def detect(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert('RGB')

    # Call the object detection function
    class_names, boxes = detect_objects(img)

    #return{'class_names': class_names, 'boxes': boxes}
    return {'class_names': class_names}

if __name__ == '__main__':
    app.run(debug=True)
