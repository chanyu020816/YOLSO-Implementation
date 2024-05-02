import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_yolo_labels(image, bboxes):
    cmap = plt.get_cmap('tab20b')

    im = np.array(image)
