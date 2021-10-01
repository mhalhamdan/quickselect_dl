import PIL
from deepface import DeepFace
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import cm

# Action = 'race' or 'gender' or 'age' or 'emotion'
def face_information(image_path, action):
    input = np.array(image_path)
    info = DeepFace.analyze(input, actions=[action])

    if action not in ['gender', 'age']:
        return max(info[action], key=info[action].get)
    else:
        return info[action]

# Input: Image in Numpy Array format
# Output: Bounding box in Dict format
def detect_face(image):
    input = np.array(image)
    result = DeepFace.analyze(input, actions=['gender'], detector_backend='opencv')

    return result['region']

def main():
    pass


if __name__ == "__main__":
    main()
  