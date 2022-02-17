from io import BytesIO
from zipfile import ZipFile
import requests
from datetime import datetime
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split

train_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
test_imgs_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
test_labels_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'

# pre-computed Histogram of Gradients (HoG) features - might be useful for more traditional approaches image recognition
train_hog_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_HOG.zip'
test_hog_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_HOG.zip'

def download_and_extract_zip(url, target_dir):
  print(f"start download from '{url}'")
  zip_bytes = requests.get(url).content
  print(f"download finished")
  z = ZipFile(BytesIO(zip_bytes))
  z.extractall(target_dir)
  print(f'extracted downloaded zip archive to {target_dir}')

# this code is adapted from http://benchmark.ini.rub.de/Dataset/GTSRB_Python_code.zip
def read_traffic_signs(rootpath: str, test_label_csv_path=None):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: 
    * path to the traffic sign data, for example './GTSRB/Training'
    * optional path to CSV file for test set Ground Truth labels (if loaded data is test set)

    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels

    if test_label_csv_path: # we should test set: all images in same folder, .csv file with labels at provided path
      with open(test_label_csv_path) as gtFile: # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(f'{rootpath}/{row[0]}')) # the 1th column is the filename
            labels.append(int(row[7])) # the 8th column is the label

    else: # training set; all images in one folder, separate .csv file with labels (expected to be in same dir)
      # loop over 42 classes (each in own subdirectory)
      for c in range(0,43):
          prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
          with open(prefix + 'GT-'+ format(c, '05d') + '.csv') as gtFile: # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                labels.append(int(row[7])) # the 8th column is the label

    return images, np.array(labels)

def resize_img(img_arr, width, height):
  img = Image.fromarray(img_arr, 'RGB')
  resized_img = img.resize((width, height))
  return np.array(resized_img)

def process_imgs(imgs, target_width, target_height):
  # resize all images to provided target width + height
  # store in 3D NumPy array
  imgs_processed = np.array([resize_img(img, target_width, target_height) for img in imgs])

  # normalize values to [0, 1] range
  max_val = imgs_processed.max()
  min_val = imgs_processed.min()
  imgs_processed= (imgs_processed - min_val)/(max_val - min_val)

  return imgs_processed

class GTSRBLoader():
  # hacky solution to implement "data cache" (prevent unnecessary downloads of the data)
  # A GTSRBLoader always creates a file with this name in self.data_path
  # GTSRBLoaders always check if a file with this name is present in self.data_path 
  DOWNLOAD_FINISHED_FILENAME = 'download_finished.txt'

  def __init__(self, data_path):
    self.data_path = data_path
    self.train_path = f'{self.data_path}/GTSRB/Final_Training/Images'
    self.test_path = f'{self.data_path}/GTSRB/Final_Test/Images'
    self.test_label_csv_path = f'{self.data_path}/GT-final_test.csv'
    self.__load_data()

  @property
  def __data_downloaded(self):
    return os.path.exists(f'{self.data_path}/{GTSRBLoader.DOWNLOAD_FINISHED_FILENAME}')

  def __load_data(self):
    if not self.__data_downloaded:
      download_and_extract_zip(train_link, self.data_path)
      download_and_extract_zip(test_labels_link, self.data_path)
      download_and_extract_zip(test_imgs_link, self.data_path)
      with open(f'{self.data_path}/{GTSRBLoader.DOWNLOAD_FINISHED_FILENAME}', 'w') as f:
        f.write(f'Data download finished {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
      print('Finished downloading required files for GTSRB dataset')
    else:
      print('GTSRB dataset already downloaded, loading files from memory')

    print('loading training images and labels')
    self.train_imgs, self.train_labels = read_traffic_signs(self.train_path)
    print('loading test images and labels')
    self.test_imgs, self.test_labels = read_traffic_signs(self.test_path, self.test_label_csv_path)
    print('done')

  def get_processed_imgs(self, target_width, target_height):
    print('processing training images')
    train_imgs_processed = process_imgs(self.train_imgs, target_width, target_height)
    print('processing test images')
    test_imgs_processed = process_imgs(self.test_imgs, target_width, target_height)
    
    print('done processing, creating train/val/test split')
    X_train, X_val, y_train, y_val = train_test_split(train_imgs_processed, self.train_labels, test_size=0.3, random_state=42, shuffle=True)
    X_test = test_imgs_processed
    y_test = self.test_labels
    
    return X_train, X_val, X_test, y_train, y_val, y_test

    

