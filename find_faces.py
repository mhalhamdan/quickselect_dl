import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from inference import run

preprocessing = transforms.ToTensor()

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
def preprocess_image(image):
    preproc_img = preprocessing(image)
    normalized_inp = preproc_img.unsqueeze(0)
    return(normalized_inp)

def main():

    image_path = './examples/example2.jpg' 

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    image = Image.open(image_path)
    input = preprocess_image(image)

    model.eval()
    output = model(input)[0]

    score_threshold = .95
    
    for box, score, label in zip(output['boxes'], output['scores'], output['labels']):

        if score > score_threshold and label == 1:
            
            print(f"This is a {COCO_NAMES[label]} (confidence:{score})")
            print("with bounding boxes: ", box)

            top, right, bottom, left = box.cpu().detach().numpy().astype(int)
            
            image = Image.open(image_path)
            # Problem in cropping dimensions caused by portrait versus non-portrait photos
            # TODO: if portrait: flip dimensions
            image = image.crop((top, right, bottom, left))
            # image.show()
            # Segment image after boxing it
            run(image)




if __name__ == "__main__":
    main()