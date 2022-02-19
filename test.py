from data_loaders import CIFAR10Loader, GTSRBLoader
import matplotlib.pyplot as plt
import numpy as np

def show_img(img_arr, ax=None):
  if ax:
    ax.imshow(img_arr)
  else:
    plt.imshow(img_arr)

def check_loaded_data(data, dataset_name):
  print(f'\n--- Checking loaded {dataset_name} data ---')
  print('data types')
  for val in data:
    print(type(val))
  print()

  print('data shapes')
  for val in data:
    shape = getattr(val, 'shape', None)
    if not (shape == None): print(val.shape)
    else: print(f'Warning: Expected property "shape", but it was not present!')
  print()

  all_imgs = np.concatenate(data[:3])
  print('Image stats:')
  print(f'shape: {all_imgs.shape}')
  print(f'Max value across all images and channels: {all_imgs.max()}')
  print(f'Min value across all images and channels: {all_imgs.min()}')

  split_sizes = [get_length(data_split) for data_split in data[:3]]
  train_size, val_size, test_size = split_sizes
  total_size = sum(split_sizes)
  print('split sizes:')
  print(f'{train_size=}, {val_size=}, {test_size=}')
  print(f'Total dataset size: {total_size}')
  print()

  split_ratios = [get_ratio_rounded(split_size, total_size) for split_size in split_sizes]
  print(f'Train/Val/Test ratio: {"/".join(str(split) for split in split_ratios)}')
  train_and_val_size = train_size + val_size
  print('Train/Val ratio:' + f"{get_ratio_rounded(train_size, train_and_val_size)}/{get_ratio_rounded(val_size, train_and_val_size)}")
  print('(Train+Val)/Test ratio:' + f"{get_ratio_rounded(train_and_val_size, total_size)}/{get_ratio_rounded(test_size, total_size)}")

  print()

index_to_data_part = {
  0: 'X_train',
  1: 'X_val',
  2: 'X_test',
  3: 'y_train',
  4: 'y_val',
  5: 'y_test'
}

def get_length(data):
  shape = getattr(data, 'shape', None)
  if not (shape == None): return data.shape[0]
  else: return len(data)

def get_ratio_rounded(part, total):
  return round(part/total, 2)

def plot_first_img_from_data(data, data_name, ax=None):
  print(f'loading {data_name}')
  print(f'shape: {data.shape}')
  img_data = data[0]

  print('\nloading first image')
  print(f'shape: {img_data.shape}\n')
  show_img(img_data, ax=ax)

CIFAR10_data = CIFAR10Loader('./data').get_processed_imgs(42, 42, normalize=False, convert_to_grayscale=True)

check_loaded_data(CIFAR10_data, 'CIFAR-10')


GTSRB_data = GTSRBLoader('./data/GTSRB').get_processed_imgs(42, 42, normalize=False, convert_to_grayscale=True)

check_loaded_data(GTSRB_data, 'GTSRB')


# check if first three return values of both datasets are valid image arrays
# we simply try to plot first image of X_train, X_val, and X_test
fig, ax = plt.subplots(2, 3)
ax = ax.flat

print('\nPlotting example images from CIFAR-10')
for i in range(3):
  plot_first_img_from_data(CIFAR10_data[i], index_to_data_part[i], ax[i])

print('\nPlotting example images from GTSRB')
for i in range(3, 6):
  data_idx = i - 3
  ax_idx = i
  plot_first_img_from_data(GTSRB_data[data_idx], index_to_data_part[data_idx], ax[ax_idx])

plt.show()