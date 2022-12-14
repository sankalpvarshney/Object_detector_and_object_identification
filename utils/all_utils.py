import cv2
import yaml

def crop_box(x, img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cropped_image = img[c1[1]:c2[1], c1[0]:c2[0]]
    cv2.imwrite("cropped_image.jpg",cropped_image)

    return cropped_image

def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content