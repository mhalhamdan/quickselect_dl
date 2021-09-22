import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from inference import run
import sys

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

def main(image_path, result_path= False):
    # TODO: Prompt user before segmenting further
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    image = Image.open(image_path)

    # If PNG, fix
    image = image.convert('RGB')

    input = preprocess_image(image)

    model.eval()
    output = model(input)[0]

    score_threshold = .95
    
    # For box count
    index = range(len(output['boxes']))

    for idx, box, score, label in zip(index, output['boxes'], output['scores'], output['labels']):

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
            # Change path depending on no. of box
            if result_path:
                if len(index) > 1:
                    r_path = result_path[0:-4] + str(idx) + result_path[-4:]
                    run(image, r_path)
                else:
                    run(image, result_path)




if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        main(sys.argv[1])
        
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])