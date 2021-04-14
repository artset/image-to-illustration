import os
import numpy as np
from PIL import Image


file_list=[]
for root, _, files in os.walk("../../data/miyazaki"):
    for name in files:
        if name.endswith(".jpg"):
            file_list.append(os.path.join(root, name))

# Import images
images = []
wrong = []
for i, file_path in enumerate(file_list):
  print("----", file_path)
  img = Image.open(file_path)
  img = np.array(img)
