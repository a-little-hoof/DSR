import os
import random
from PIL import Image
import torch

def load_images(imagenet_path, count = 10, images_only = True):
  # Sample random images from imagenet
  file_list = []

  for root, dirs, files in os.walk(imagenet_path):
      for file in files:
          file_list.append(os.path.join(root, file))

  if count > len(file_list):
    sampled_files = file_list
  else:
    sampled_files = random.sample(file_list, count)

  image_files = []

  for filename in sampled_files:
    image_files.append(Image.open(filename))
  print("Loaded {} images".format(len(image_files)))

  if images_only:
    return image_files
  else:
    return image_files, sampled_files

def filter_highest_layer(register_norms, highest_layer):
  return [norm for norm in register_norms if norm[0] <= highest_layer]

def filter_lowest_layer(register_norms, lowest_layer):
  return [norm for norm in register_norms if norm[0] >= lowest_layer]

def filter_layers(register_norms, highest_layer = -1, lowest_layer = 0):
  if highest_layer == -1:
    return [norm for norm in register_norms if norm[0] >= lowest_layer]
  else:
    return [norm for norm in register_norms if norm[0] >= lowest_layer and norm[0] <= highest_layer]

def sign_max(tensor):
   pos_max = torch.max(tensor)
   neg_max = torch.min(tensor)
   return pos_max if abs(pos_max) > abs(neg_max) else neg_max