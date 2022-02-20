import matplotlib.pyplot as plt

def plot_loss_and_acc(hist, epochs):
  '''
  plots loss and accuracy from Keras history object (returned after calling .fit() on a model)
  '''
  colors = {'loss':'r', 'accuracy':'b', 'val_loss':'m', 'val_accuracy':'g'}
  plt.figure(figsize=(10,6))
  plt.title("Training Curve") 
  plt.xlabel("Epoch")

  for measure in hist.keys():
      color = colors[measure]
      plt.plot(range(1,epochs+1), hist[measure], color + '-', label=measure)  # use last 2 values to draw line

  plt.legend(loc='upper left', scatterpoints = 1, frameon=False)
  plt.show()