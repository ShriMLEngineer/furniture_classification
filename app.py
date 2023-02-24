from io import BytesIO
import base64
import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, jsonify, request

# load pre-trained DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# define the API endpoint
app = Flask(__name__)

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # get the image data from the request
    image_data = request.get_json()['image']
    # convert the image data from base64 to bytes
    image_bytes = base64.b64decode(image_data)
    # open the image using PIL
    image = Image.open(BytesIO(image_bytes))
    # apply the necessary transforms to the image
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    # add batch dimension to the image
    image = image.unsqueeze(0)
    # run the image through the DETR model
    outputs = model(image)
    # get the predicted classes for each object
    pred_classes = [int(x) for x in outputs['pred_classes'][0]]
    # return the predicted classes as a JSON response
    response = {
        'status': 'success',
        'objects': pred_classes
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
