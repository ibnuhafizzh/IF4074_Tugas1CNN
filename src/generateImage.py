from PIL import Image
from matplotlib.patheffects import withSimplePatchShadow
import numpy as np
import os
import cv2

# function to read image to numpy array
def read_dataset(dataset_path):
    list_folder_per_kelas = os.listdir(dataset_path)
    list_folder_per_kelas = sorted(list_folder_per_kelas)
    file_path = []; class_label = np.ndarray(shape=(0)); class_dictionary = {}
    
    for i in range(1, len(list_folder_per_kelas)):#loop for all class folders
        class_folder_path = os.path.join(dataset_path, list_folder_per_kelas[i])
        list_image_name = os.listdir(class_folder_path)
        list_image_name = sorted(list_image_name)
        temp_file_path = [os.path.join(class_folder_path, j) for j in list_image_name]
        file_path += temp_file_path
        temp_class_label = np.full((len(list_image_name)),np.int16(i))
        class_label = np.concatenate((class_label, temp_class_label), axis=0)
        class_dictionary[str(i)] = list_folder_per_kelas[i]
    return np.asarray(file_path), class_label, class_dictionary


def list_img_to_matrix(img_path, size = (400, 400)):
    prepocessed_images = []
    for file in img_path:
        img = cv2.imread(file, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_final = cv2.resize(img, size)
        
        img_final = np.array(img_final)
        prepocessed_images.append(img_final)
    
    prepocessed_images = np.array(prepocessed_images, dtype="object")
    prepocessed_images = np.transpose(prepocessed_images, (0,3,1,2))
    return prepocessed_images


