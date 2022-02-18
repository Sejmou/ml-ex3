from data_loaders import CIFAR10Loader, GTSRBLoader
import matplotlib.pyplot as plt

def show_img(img_arr, ax=None):
  if ax:
    ax.imshow(img_arr)
  else:
    plt.imshow(img_arr)

def check_returned_data(data, dataset_name):
  print(f'--- Checking returned {dataset_name} data ---')
  print('data types')
  for val in data:
    print(type(val))

  print('data shapes')
  for val in data:
    shape = getattr(val, 'shape', None)
    if not (shape == None): print(val.shape)
    else: print(f'Warning: Expected property "shape", but it was not present!')

  #print('Train/Val/Test ratio:')

CIFAR10_data = CIFAR10Loader('./data').get_processed_imgs(42, 42)

check_returned_data(CIFAR10_data, 'CIFAR-10')


GTSRB_data = GTSRBLoader('./data/GTSRB').get_processed_imgs(42, 42)

check_returned_data(GTSRB_data, 'GTSRB')


# check if first three return values of both datasets are valid image arrays
# we simply try to plot first image of X_train, X_val, and X_test
fig, ax = plt.subplots(3, 2)
ax = ax.flat

for i in range(3):
  show_img(CIFAR10_data[i][0])

for i in range(3, 6):
  show_img(GTSRB_data[i][0])
