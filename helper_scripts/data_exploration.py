import matplotlib.pyplot as plt
import numpy as np

def show_img(img_arr, ax=None):
  grayscale = img_arr.ndim == 2
  if ax:
    if grayscale:
      ax.imshow(img_arr, cmap='gray', vmin=0, vmax=255)#https://stackoverflow.com/a/3823822/13727176
    else: 
      ax.imshow(img_arr)
  else:
    if grayscale:
      plt.imshow(img_arr, cmap='gray', vmin=0, vmax=255)
    else: 
      plt.imshow(img_arr)

#maps indices of data returned from data loader to text description of the respective data
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

def print_dataset_summary(data, dataset_name):
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