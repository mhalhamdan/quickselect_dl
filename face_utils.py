from deepface import DeepFace
import numpy as np
import pandas as pd
from PIL import Image

# Action = 'race' or 'gender' or 'age' or 'emotion'
def face_information(image_path, action):
    input = np.array(image_path)
    info = DeepFace.analyze(input, actions=[action])

    if action not in ['gender', 'age']:
        return max(info[action], key=info[action].get)
    else:
        return info[action]

# Input: PIL image
# Output: Bounding box in Dict format
def detect_face(image):
    input = np.array(image)
    result = DeepFace.analyze(input, actions=['gender'], detector_backend='opencv')

    return result['region']

# Input: PIL image
# Output cropped PIL image
def find_face(image):
    result = detect_face(image)

    FACTOR = 1.5
    x = result['x']/FACTOR
    y = result['y']/(FACTOR+1.5)
    w = result['w']*FACTOR
    h = result['h']*(FACTOR)

    result = image.crop((x, y, w+x, h+y))
    return result

def main():
    pass


if __name__ == "__main__":
    main()
  