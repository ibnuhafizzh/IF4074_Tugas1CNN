from PIL import Image
import numpy as np
import os
import cv2

DATA = "/Users/mac/Documents/ITB/Semester7/ML/IF4074_Tugas1CNN/test"

# function to read image to numpy array
def read_dataset(dataset_path):
    """
    parameter:
    (a) dataset_path: path of the dataset [type: string]
    
    this function will return:
    (1) file_path: full path of each image file [type: 1D numpy array]
    (2) class_label: class of each image file, in numerical value 
                     [type: 1D numpy array]
    (3) class_dictionary: key(string) -> value (string), where 'key' is from (2) 
        and 'value' is from 'folder_name' [type: dictionary]
    """
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


def img_to_matrix(img_path):
    '''
    parameter:
    a. img_path: file_path, which is 1D numpy array of all image file paths 
                 in the "Given Basic Code".
    b. size: tupple of desired image sizes after zero padding, 
             which is (height, width). In this hands-on week, 
             use the default value of (100, 100).
    return:
    a. prepocessed_images = 4D numpy array with the size of 
                            (150,  desired_height, desired_width, 3).
    HINT: use "cv2.resize()" API so that the longer dimension becomes 100,
           then, pad the shorter dimension with zero intensity values.
    Tasks:
    a. prepocessed_images is in RGB format
    b. Please show first-two prepocessed_images with their sizes. 
       Make sure that both sizes are same.
    '''
    prepocessed_images = []
    for file in img_path:
        img = cv2.imread(file, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_final = cv2.resize(img, (100, 100))
        
        img_final = np.array(img_final)
        prepocessed_images.append(img_final)
    
    prepocessed_images = np.array(prepocessed_images)
    return prepocessed_images

dataset_path = DATA
file_path, class_label, class_dictionary = read_dataset(dataset_path)
print("file_path:\n", file_path[0:len(file_path)], ", shape:", file_path.shape)
print("\nclass_label:\n", class_label[0:len(class_label)], ", shape:", class_label.shape)
print("\nclass_dictionary:\n", class_dictionary)
print("\nmatrix imagee:\n", img_to_matrix(file_path))