from ImageStructureTensor import ImageStructureTensor
from VisualizationTools import VisualizationTools
import numpy as np
import pickle
import matplotlib.pyplot as plt

structure_tensor = ImageStructureTensor("coins_mask.jpg","Sobel")
structure_tensor.computeStructureTensor()
visual_tools = VisualizationTools()
visual_tools.visualize_eigenvector(structure_tensor.img,structure_tensor.eigValues[:,:,0],structure_tensor.eigVectors[:,:,:,0],4)