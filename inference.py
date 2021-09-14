import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

import time

resize = transforms.Resize(640)
preprocessing = transforms.Compose([transforms.Resize(640), 
                                    transforms.ToTensor()])
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

model_dict = {
    1: 'deeplabv3_resnet50',
    2: 'deeplabv3_resnet101',
    3: 'fcn_resnet50',
    4: 'fcn_resnet101',
    5: 'deeplabv3_mobilenet_v3_large',
    6: 'lraspp_mobilenet_v3_large'}

def initialize_model(name='deeplabv3'):
    if name == 'deeplabv3_resnet50':
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)

    if name == 'deeplabv3_resnet101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)

    if name == 'fcn_resnet50':
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)

    if name == 'fcn_resnet101':
        model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)

    if name == 'deeplabv3_mobilenet_v3_large':
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)

    if name == 'lraspp_mobilenet_v3_large':
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)

    return model

def preprocess_image(path):
    image = Image.open(path)
    preproc_img = preprocessing(image)
    normalized_inp = normalize(preproc_img).unsqueeze(0)
    normalized_inp.requires_grad = True
    return(normalized_inp)

def make_prediction(model, image):
    input = preprocess_image(image)
    model = model.eval()
    output = model(input)['out']
    # print(output.shape, output.min().item(), output.max().item())
    return output

def visualize_prediction(output, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(output).astype(np.uint8)
    g = np.zeros_like(output).astype(np.uint8)
    b = np.zeros_like(output).astype(np.uint8)

    # white = (255,255,255)
    # idx = output == 15
    # r[idx] = 0
    # g[idx] = 0
    # b[idx] = 0

    idx = output == 0
    r[idx] = 255
    g[idx] = 255
    b[idx] = 255

    rgb = np.stack([r, g, b], axis=2)

    return rgb

def full_image(image, mask):

    trans = transforms.ToPILImage()
    mask = trans(mask)
    mask = mask.convert("L")
    
    image = resize(image)

    background = Image.new("RGBA", image.size, 0)
    draw = ImageDraw.Draw(background)
    print(image.size)
    draw.rectangle((image.size, image.size), fill=255)

    result = Image.composite(background, image, mask)
    return result



def test():

    image = './sample.jpg'

    for model_name in model_dict.values():
        model = initialize_model(model_name)

        start = time.time()
        output = make_prediction(model, image)
        end = time.time()
        print(f"Inference process took {end - start} seconds using model: {model_name}")
        

def main():
    # Test case
    model = initialize_model(model_dict[5])
    image = './sample.jpg'
    output = make_prediction(model, image)

    # Find most likely segmentation class for each pixel.
    out_max = torch.argmax(output, dim=1, keepdim=True)
    rgb = visualize_prediction(out_max.detach().cpu().squeeze().numpy())
    # plt.imshow(rgb); plt.axis('off'); plt.show()

    # Test mask applied on full_image
    image = Image.open('./sample.jpg')
    # image = preprocessing(image)

    final = full_image(image, rgb)
    final.save('./test2.png')
    plt.imshow(final); plt.axis('off'); plt.show()




if __name__ == "__main__":
    main()
    # test()