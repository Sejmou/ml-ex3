import matplotlib.pyplot as plt

from notebooks.helper_scripts.data_loaders import CIFAR10Loader, GTSRBLoader
from notebooks.helper_scripts.data_exploration import print_dataset_summary, show_img, data_idx_to_desc, file_and_folder_overview

file_and_folder_overview('.')

print(f'CIFAR-10 labels:')
print(CIFAR10Loader.TEXT_LABEL_DICT)
print()
print(f'GTSRB labels')
print(GTSRBLoader.TEXT_LABEL_DICT)
print()

CIFAR10_loader = CIFAR10Loader('./data')
CIFAR10_data = CIFAR10_loader.get_processed_imgs(42, 42, normalize=False, convert_to_grayscale=True)
GTSRB_loader =GTSRBLoader('./data/GTSRB')
GTSRB_data = GTSRB_loader.get_processed_imgs(42, 42, normalize=False, convert_to_grayscale=True)

print(f'CIFAR-10 labels from instance:')
print(CIFAR10Loader.TEXT_LABEL_DICT)
print(CIFAR10_loader.text_label_dict)

print()
print(f'GTSRB labels from instance')
print(GTSRBLoader.TEXT_LABEL_DICT)
print(GTSRB_loader.text_label_dict)


print_dataset_summary(CIFAR10_data, 'CIFAR-10')
print_dataset_summary(GTSRB_data, 'GTSRB')


# check if first three return values of both datasets are valid image arrays
# we simply try to plot first image of X_train, X_val, and X_test
fig, ax = plt.subplots(2, 3)
ax = ax.flat

def plot_first_img_from_data(data, data_name, ax=None):
  print(f'loading {data_name}')
  print(f'shape: {data.shape}')
  img_data = data[0]

  print('\nloading first image')
  print(f'shape: {img_data.shape}\n')
  show_img(img_data, ax=ax)

print('\nPlotting example images from CIFAR-10')
for i in range(3):
  plot_first_img_from_data(CIFAR10_data[i], data_idx_to_desc[i], ax[i])

print('\nPlotting example images from GTSRB')
for i in range(3, 6):
  data_idx = i - 3
  ax_idx = i
  plot_first_img_from_data(GTSRB_data[data_idx], data_idx_to_desc[data_idx], ax[ax_idx])

plt.show()