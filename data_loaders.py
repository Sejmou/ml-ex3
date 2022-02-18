from io import BytesIO
from zipfile import ZipFile
import requests
import tarfile
import pickle
import glob
from datetime import datetime
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split


def download_and_extract_zip(url, target_dir):
  print(f"start download from '{url}'")
  zip_bytes = requests.get(url).content
  print(f"download finished")
  z = ZipFile(BytesIO(zip_bytes))
  z.extractall(target_dir)
  print(f'extracted downloaded zip archive to {target_dir}')

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

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

class BaseDataLoader():
  # hacky solution to implement "data cache" (prevent unnecessary downloads of the data)
  # A DataLoader always creates a file with this name in self._data_path
  # DataLoaders always check if a file with this name is present in self._data_path 
  DOWNLOAD_FINISHED_FILENAME = 'download_finished.txt'# may be overridden in subclassees
  DATASET_NAME = 'Data'# may also be overridden

  def __init__(self, data_path):
    self._data_path = data_path

    # note: the following properties must all be set to meaningful values in inheriting/implementing classes!
    self.train_imgs = np.array([])
    self.train_labels = np.array([])
    self.test_imgs = np.array([])
    self.test_labels = np.array([])

    self._load_data()

  @property
  def _data_downloaded(self):
    return os.path.exists(f'{self._data_path}/{self.DOWNLOAD_FINISHED_FILENAME}')

  def _confirm_successful_download(self):
    with open(f'{self._data_path}/{self.DOWNLOAD_FINISHED_FILENAME}', 'w') as f:
      f.write(f'\Data download finished {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    print(f'Finished downloading required files for {self.DATASET_NAME} dataset')

  def _load_data(self):
    """Retrieves data and stores it in object properties (if data already present locally, loads those files, else downloads data from the web)
    """
    raise NotImplementedError()#TODO: could play around with abstract base classes (newer Python feature, abc module), probably completely overkill!

  def get_processed_imgs(self, target_width, target_height):
    """Returns images + labels of the dataset in a train/test/val split.

     X_train, X_test, X_val, y_train, y_test, y_val are all NumPy arrays.

     X_... are 4D arrays of images (shape: n x image_width x image_height x no. of color channels)
     
     y_... are 1D arrays of numeric labels
    """

    print('processing training images')
    train_imgs_processed = process_imgs(self.train_imgs, target_width, target_height)
    print('processing test images')
    test_imgs_processed = process_imgs(self.test_imgs, target_width, target_height)
    
    print('done processing, creating train/val/test split')
    X_train, X_val, y_train, y_val = train_test_split(train_imgs_processed, self.train_labels, test_size=0.3, random_state=42, shuffle=True)
    X_test = test_imgs_processed
    y_test = self.test_labels
    
    return X_train, X_val, X_test, y_train, y_val, y_test

class GTSRBLoader(BaseDataLoader):
  DOWNLOAD_FINISHED_FILENAME = 'download_finished.txt'
  DATASET_NAME = 'GTSRB'

  train_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip'
  test_imgs_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip'
  test_labels_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'
  # pre-computed Histogram of Gradients (HoG) features - might be useful for more traditional approaches image recognition
  train_hog_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_HOG.zip'
  test_hog_link = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_HOG.zip'

  def __init__(self, data_path):
    super(GTSRBLoader, self).__init__(data_path)

  def _load_data(self):
    train_path = f'{self._data_path}/GTSRB/Final_Training/Images'
    test_path = f'{self._data_path}/GTSRB/Final_Test/Images'
    test_label_csv_path = f'{self._data_path}/GT-final_test.csv'

    if not self._data_downloaded:
      download_and_extract_zip(self.train_link, self._data_path)
      download_and_extract_zip(self.test_labels_link, self._data_path)
      download_and_extract_zip(self.test_imgs_link, self._data_path)
      self._confirm_successful_download()
    else:
      print('GTSRB dataset already downloaded, loading files from memory')

    print('loading training images and labels')
    self.train_imgs, self.train_labels = GTSRBLoader._read_traffic_signs(train_path)
    print('loading test images and labels')
    self.test_imgs, self.test_labels = GTSRBLoader._read_traffic_signs(test_path, test_label_csv_path)
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

  # this code is adapted from http://benchmark.ini.rub.de/Dataset/GTSRB_Python_code.zip
  @staticmethod
  def _read_traffic_signs(rootpath: str, test_label_csv_path=None):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: 
    * path to the traffic sign data, for example './GTSRB/Training'
    * optional path to CSV file for test set Ground Truth labels (if loaded data is test set)

    Returns:   list of images (note: sizes vary!), list of corresponding labels'''
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
          #print(f'loading images for class {c}')
          prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
          with open(prefix + 'GT-'+ format(c, '05d') + '.csv') as gtFile: # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                labels.append(int(row[7])) # the 8th column is the label

    return images, np.array(labels)

    


class CIFAR10Loader(BaseDataLoader):
  DATASET_NAME = 'CIFAR-10'

  def __init__(self, data_path):
    super(CIFAR10Loader, self).__init__(data_path)

  def _load_data(self):
    if not self._data_downloaded:
      archive_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
      print(f'Downloading {self.DATASET_NAME} data from {archive_url}')
      response = requests.get(archive_url, stream=True)
      file = tarfile.open(fileobj=response.raw, mode="r|gz")
      file.extractall(self._data_path)
      self._confirm_successful_download()
    else:
      print('CIFAR-10 dataset already downloaded, loading files from memory')
    
    #The extracted archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch.
    #Each of these files is a Python "pickled" object produced with cPickle.
    batch_filenames = glob.glob(f'{self._data_path}/cifar-10-batches-py/*_batch*')

    test_batch_filename = next(filter(lambda filename: 'test' in filename, batch_filenames), None)
    if not test_batch_filename:
      raise RuntimeError(f'No test batch found in {self._data_path}!')
    
    train_batch_filenames = list(filter(lambda filename: not filename == test_batch_filename, batch_filenames))
    if not train_batch_filenames:
      raise RuntimeError(f'No train batches found in {self._data_path}!')

    # The next steps are unpickling the data batches and creating the training + testing data from them

    #unpickling returns dictionaries with four keys:
    # * batch_label -- describes the "name"/purpose of the batch (may be one of the 5 test batches or the single test batch)
    # * data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # * labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    # * filenames -- a list storing the filenames of each respective image in data -> irrelevant to us!
    
    # we will need to put all the data and labels from the training batches together into NumPy arrays of shapes n x 32 x 32 x 3 and n.
    # Note that we therefore also need to reshape the image data and concatenate all arrays!
    # A similar process can be followed for the test data and labels, however we just use the single batch and don't need to concatenate.

    print('loading training images and labels')
    train_batches = [unpickle(filename) for filename in train_batch_filenames]
    train_imgs, train_labels = zip(*[CIFAR10Loader._get_batch_imgs_and_labels(batch) for batch in train_batches])
    self.train_imgs = np.concatenate(train_imgs)
    self.train_labels = np.concatenate(train_labels)

    print('loading test images and labels')
    test_batch = unpickle(test_batch_filename)
    self.test_imgs, self.test_labels = CIFAR10Loader._get_batch_imgs_and_labels(test_batch)
    print('done')

  @staticmethod
  def _get_batch_imgs_and_labels(batch):
    # note: each batch is a dictionary, its keys are byte strings!
    data = batch.get(b'data')
    imgs = CIFAR10Loader._reshape_batch_data(data)
    labels = np.array(batch.get(b'labels'))
    return imgs, labels

  @staticmethod
  def _reshape_batch_data(batch_data):
    '''
    Reshapes the batch data from n x 3072 to n x 32 x 32 x 3
    '''

    #"problem": batch data has (is expected to have) shape n x 3072
    # * n is the number of images in the batch
    # * 3072 are the pixels of each image (1024 per channel)
    # We know that the CIFAR-10 dataset contains 32 x 32 images, and 32 x 32 happens to be exactly 1024 :)
    # our goal is to get a matrix of shape (n, 32, 32, 3), where 3 is the number of color channels
    n = batch_data.shape[0]
    res = batch_data.reshape(n, 3, 32, 32)

    # Now, we need to "reaarange" the dimensions using NumPy's transpose() - see https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html#numpy.ndarray.transpose
    # i in the j-th place means res's i-th axis becomes res.transpose()'s j-th axis.
    # so, the below call gives us exactly the desired shape: (n, 32, 32, 3)!
    res = res.transpose(0,2,3,1)
    return res