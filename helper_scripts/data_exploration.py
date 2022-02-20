import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

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
data_idx_to_desc = {
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
  if len(data) != 6:
    raise ValueError('Expected tuple of 6 values (X_train, X_val, X_test, y_train, y_val, y_test) as argument for "data"')

  print(f'\n--- Checking loaded {dataset_name} data ---')
  print('Data types')
  for i, val in enumerate(data):
    print(f'{data_idx_to_desc[i]}: {type(val)}')
  print()

  print('Data shapes')
  for i, val in enumerate(data):
    shape = getattr(val, 'shape', None)
    if not (shape == None): print(f'{data_idx_to_desc[i]}: {shape}')
    else: print(f'{data_idx_to_desc[i]}: Expected property "shape", but it was not present! len() is {len(val)}')
  print()

  all_imgs = np.concatenate(data[:3])
  print('Image stats:')
  print(f'shape: {all_imgs.shape}')
  print(f'Max value across all images and channels: {all_imgs.max()}')
  print(f'Min value across all images and channels: {all_imgs.min()}')

  split_sizes = [get_length(data_split) for data_split in data[:3]]
  train_size, val_size, test_size = split_sizes
  total_size = sum(split_sizes)
  print('Split sizes:')
  print(f' train size: {train_size}\nvalidation size: {val_size}\ntest size: {test_size}')
  print(f'Total dataset size: {total_size}')
  print()

  split_ratios = [get_ratio_rounded(split_size, total_size) for split_size in split_sizes]
  print(f'Train/Val/Test ratio: {"/".join(str(split) for split in split_ratios)}')
  train_and_val_size = train_size + val_size
  print('Train/Val ratio:' + f"{get_ratio_rounded(train_size, train_and_val_size)}/{get_ratio_rounded(val_size, train_and_val_size)}")
  print('(Train+Val)/Test ratio:' + f"{get_ratio_rounded(train_and_val_size, total_size)}/{get_ratio_rounded(test_size, total_size)}")



def file_and_folder_overview(startpath, exclude_hidden = True):# adapted from https://stackoverflow.com/a/9728478/13727176
  '''
  Helper function for getting overview of files/folders within given startpath (incl. subfolders and containing files)

  Prints the tree of (sub-)directories and files in recusive fashion, starting from startpath.
  For output readability reasons, at max. 5 files of the same file extension are displayed (the number of remaining files of the same file extension is also printed)
  '''

  def get_file_extension_and_name(filename):
    file_and_extension = filename.split(sep='.')
    extension = file_and_extension[-1] if len(file_and_extension) > 1 else '(No extension)'
    return extension, filename

  for root, dirs, files in os.walk(startpath):
    if exclude_hidden:
      if '.git' in root: #content of .git is not very interesting for our purposes...
        continue
      # https://stackoverflow.com/a/13454267/13727176
      files = [f for f in files if not f[0] == '.']
      dirs[:] = [d for d in dirs if not d[0] == '.']

    dirs = dirs.sort() 

    files_by_extension = defaultdict(list)
    for ext, filename in [get_file_extension_and_name(f) for f in files]:
      files_by_extension[ext].append(filename)

    level = root.replace(startpath, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print('{}{}/'.format(indent, os.path.basename(root)))

    subindent = ' ' * 4 * (level + 1)

    for ext, files in files_by_extension.items():
      filtered_files = files[:5]
      
      for f in filtered_files:
        print('{}{}'.format(subindent, f))
      if len(files) > len(filtered_files):
        remaining_files = len(files) - len(filtered_files)
        more_text = f'... and {remaining_files} other .{ext} files'
        print('{}{}'.format(subindent, more_text))