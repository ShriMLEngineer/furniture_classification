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
      
 
   - **Model Inferencing:**<br />
    - Model inferencing code is present in model_training.ipynb
    
   - **Graphs of mAP, Loss and Class error:**
      Model was training for 20 epochs and below are graphs of training
      
      ![image](https://user-images.githubusercontent.com/126147358/221084155-461cb99a-3fd3-4fb5-b150-0f43a898c124.png)

   - **Model testing:**
      Model was tested on 15 images which were NOT part of training. Code is present in model_training.ipynb towards the end of the file
      <br />
      ![image](https://user-images.githubusercontent.com/126147358/221084805-5c82126b-a1ce-451f-991f-42d6df0bc2ab.png)
      ![image](https://user-images.githubusercontent.com/126147358/221084936-627cb91c-9f0c-4d2f-ab60-e86e01169491.png)
      ![image](https://user-images.githubusercontent.com/126147358/221084981-39157c05-2669-474d-bf8d-93a66d8dfa51.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085024-3ac28f54-dcac-4351-8b67-f71ca82f9811.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085123-02257b1a-d647-4a75-9d76-8842a87af028.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085149-315100ab-c5d9-43e8-8cd9-f8d11b9814fa.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085170-7dd56f0f-3019-4e40-bb77-8496c2cc66d0.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085199-e850ac17-52cc-43df-9521-2773853fa259.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085235-7ebae4a6-838a-4d9a-9b99-163fe3d00db3.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085263-77942efe-99b4-4202-a964-b40c262e04b5.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085282-2c3026f8-7bf1-4712-9138-3013ef29a0ea.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085317-9e8bd387-6a0d-4740-adeb-441eaae5c2ce.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085351-50a44e34-41f8-4e77-8316-4df967e1bee3.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085374-dea18bac-6a3f-4415-aa9e-b04c96eddd77.png)
      ![image](https://user-images.githubusercontent.com/126147358/221085403-cc18733d-ffd3-4a25-b19f-381bc446632c.png)









   
    
    
  
