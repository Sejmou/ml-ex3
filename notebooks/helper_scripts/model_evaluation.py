from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_test, y_pred, figsize=(16, 12), text_labels=None):
  cf = confusion_matrix(y_test, y_pred)
  df_cm = pd.DataFrame(cf, index = text_labels, columns = text_labels)
  plt.figure(figsize = figsize)
  sns.heatmap(df_cm, annot=True, fmt='g')