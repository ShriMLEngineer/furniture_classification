# furniture_classification

### Problem Statement:
**Build a classficiation model based on below conditions:**
  1. The Dataset folder has images of 3 classes. 
  2. Build a classification model (Deep learning preferred).
  3. Build an API to access the model. The API should accept an image as input and return  the predicted label (category) as output  (Preferred frameworks based on Python).
  4. Create a Docker image of your code by following docker best practices.
  5. Implement CI/CD pipeline on Github Actions.
  6. Add a clear README file with instructions.
  
### Solution:
  An approach was decided to use transformer model which can be used in google colab and can give good precision. Based on my previous experience of using transformer model in production project, DETR (Detection Transformer by Facebook) was selected for model development

  **Data Preprocessing:** <br />
  <br />
      - Total 300 images are present in dataset (100 per classes) without annotations. This data was manually split as 75%:20%:5% between train, validation and test sets. <br /> 
      - VGG Image Annotator (VIA) tool was used to annotate data in COCO format <br />
      - Once annotations were created, they were stored in different folders as mentioned below: <br />
  
   - **annotations** <br />
          - *instances_train2017.json*      This is coco annotation for training data <br />
          - *instances_val2017.json*        This is coco annotation for validation data<br />
   - **train2017**                         This folder has training images<br />
   - **val2017**                           This folder has validation images<br />

  **Budiling Classification Model:**
  
   - DETR comes up with different resnet backbones like R50, R101. For model training, R50 was selected
   - Code to invoke DETR in colab and using the same can be found in model_training.ipynb file. Link of the same is given below:
   https://github.com/ShriMLEngineer/furniture_classification/blob/main/model_training.ipynb
   - **Changes in the code:** <br />
      Below changes were made in the DETR code to customize the same as per our requirement
        - **main.py:**<br />
            Default implementation of DETR is for 91 COCO classes. Hence code has hard coded the number of classes. This hardcoding was removed and new parameter num_classes was added in main.py <br />
            `parser.add_argument('--num_classes', default=91, type=int)`
          
        - **hubconf.py:**<br />
            Code was updated to accept the num_classes passed from main.py. Hence functions *_make_detr* and *detr_resnet50* were updated to accept num_classes argument rather than using default 91 classes<br />
            <br />
            **_make_detr**<br />
              Code commented `#def _make_detr(backbone_name: str, dilation=False, num_classes=91, mask=False):` <br />
              Code added `def _make_detr(backbone_name: str, num_classes: int, dilation=False, mask=False):`<br />
            
            **detr_resnet50**<br />
              Code commented `#def detr_resnet50(pretrained=False, num_classes=91, return_postprocessor=False):` <br />
              Code added `def detr_resnet50(num_classes: int, pretrained=False, return_postprocessor=False):`<br />
              
              
   - Model training:<br />
    - Model was trained for 20 epochs with LR of 0.0001. Value 4 was passed to num_classes parameter as DETR expects 1 additional class as N/A.
    - Below is model training script
   
   `!python main.py  --coco_path "/content/furniture_classification/Dataset"  --output_dir "/content/furniture_classification/outputs" --resume "/content/furniture_classification/detr-main/detr_r50_no-class-head.pth" --num_classes=4  --epochs=20  --lr=1e-4  --batch_size=1  --num_workers=1`
      
 
   - Model training:<br />
   
    
    
  
