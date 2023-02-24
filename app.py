import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image

app = FastAPI()

# Load the trained model
path = '/content/furniture_classification/detr-main'

model = torch.hub.load(path, 'detr_resnet50', source="local", pretrained=True, num_classes=4)
checkpoint = torch.load("/content/furniture_classification/outputs/checkpoint.pth", map_location='cpu')

model.load_state_dict(checkpoint['model'], strict=False)
model.eval();

# Define the prediction function
def predict(image):
    # Load the image
    img = Image.open(image.file).convert('RGB')
    # Resize the image to the size required by the model
    img = img.resize((224, 224))
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Scale the image pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    # Add a batch dimension to the image
    img_array = np.expand_dims(img_array, axis=0)
    # Make the prediction
    prediction = model.predict(img_array)
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Define the API endpoint
@app.post('/predict')
async def predict_image(image: UploadFile = File(...)):
    # Call the prediction function
    predicted_class = predict(image)
    return {'class': predicted_class}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)