from deepface import DeepFace

# Action = 'race' or 'gender' or 'age' or 'emotion'
def face_information(image_path, action):
    info = DeepFace.analyze(image_path, actions=[action])

    if action not in ['gender', 'age']:
        return max(info[action], key=info[action].get)
    else:
        return info[action]


def main():
    pass
    # info = face_information("examples/example2.jpg", 'age')
    # print(info)


if __name__ == "__main__":
    main()
  