import os
import copy
import random
import io
import string
import numpy as np
import pandas as pd
import uuid
import flask
import urllib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
from flask import Flask , render_template  , request , send_file


import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model = torch.load(os.path.join(BASE_DIR , 'resnet18_2.pth'), map_location=torch.device('cpu'))
# model = torch.load(os.path.join(BASE_DIR , 'resnet50_2.pth'), map_location=torch.device('cpu'))
model = torch.load(os.path.join(BASE_DIR , 'mobilenet_2.pth'), map_location=torch.device('cpu'))
model.eval()

def read_csv_annotation(path, incorrect_labels_percent=0):
    if incorrect_labels_percent not in [0, 1, 5, 25, 50]:
        raise ValueError('incorrect_labels_percent should be in [0, 1, 5, 25, 50]')

    labels_column_name = f'noisy_labels_{incorrect_labels_percent}'
    annotation = pd.read_csv(path)

    annotation = annotation[['path', labels_column_name, 'is_valid']]
    annotation.rename(columns={labels_column_name: 'label'}, inplace=True)

    return annotation

dataset_annotation = read_csv_annotation(os.path.join(BASE_DIR , 'noisy_imagewoof.csv'), incorrect_labels_percent=0)
label_binarizer = LabelBinarizer().fit(dataset_annotation['label'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

classes_name = ['Shih-Tzu','Rhodesian ridgeback','Beagle','English foxhound',\
                'Australian terrier','Border terrier',\
                'Golden retriever','Old English sheepdog','Samoyed','Dingo']


def predict(filename , model):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    input_tensor = preprocess(input_image)

    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0) 

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        image_tensor = torch.FloatTensor(np.array(input_tensor)).unsqueeze(0).to(device)
        predictions = model(image_tensor).cpu().numpy()
        pred_classes = label_binarizer.inverse_transform(predictions, threshold=0)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    output = torch.nn.functional.softmax(output[0], dim=0)
    confidence, index = torch.max(output, 0)
    # top_p= torch.topk(output, 3)
    top_p, ind = output.topk(3)  
    prob = top_p.tolist()
    class_ind = ind.tolist()

    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append(round(prob[i]*100, 3))
    for i in class_ind:
        class_result.append(classes_name[i])
    return class_result , prob_result




@app.route('/')
def home():
        return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                # img_path = file.filename

                img = file.filename

                class_result , prob_result = predict(img_path , model)

                predictions = {
                      "class1":class_result[0],
                        "class2":class_result[1],
                        "class3":class_result[2],
                        "prob1": prob_result[0],
                        "prob2": prob_result[1],
                        "prob3": prob_result[2],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = True)