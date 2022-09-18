from PIL import Image
from matplotlib.patheffects import withSimplePatchShadow
import numpy as np
import os
import cv2

# function to read image to numpy array of file name
def read_dataset(dataset_path):
    folder_list = os.listdir(dataset_path)
    folder_list = sorted(folder_list)
    folder_path = []; class_label = np.ndarray(shape=(0)); class_dictionary = {}
    
    for i in range(1, len(folder_list)):#loop for all class folders
        class_folder_path = os.path.join(dataset_path, folder_list[i])
        list_image_name = os.listdir(class_folder_path)
        list_image_name = sorted(list_image_name)
        temp_folder_path = [os.path.join(class_folder_path, j) for j in list_image_name]
        folder_path += temp_folder_path
        temp_class_label = np.full((len(list_image_name)),np.int16(i))
        class_label = np.concatenate((class_label, temp_class_label), axis=0)
        class_dictionary[str(i)] = folder_list[i]
    return np.asarray(folder_path), class_label, class_dictionary

# function to read all images in a folder to numpy array of image matrix
def list_img_to_matrix(folder_path, size = (400, 400)):
    list_of_image_matrix = []
    for file_img in folder_path:
        image = cv2.imread(file_img, 1)

        image_matrix = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_matrix = cv2.resize(image_matrix, size)
        image_matrix = np.array(image_matrix)

        list_of_image_matrix.append(image_matrix)
    
    list_of_image_matrix = np.array(list_of_image_matrix, dtype="object")
    list_of_image_matrix = np.transpose(list_of_image_matrix, (0,3,1,2))
    return list_of_image_matrix


