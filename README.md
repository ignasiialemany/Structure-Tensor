# Structure Tensor

An image processing tool to obtain the structure tensor of an image. The structure tensor provides us information on the maximum pixel change
and can be used to compare different images or study certain orientations. It has been used to compare cardiovascular histology
images and obtain preferred sheetlet directions.

A clear example on how to utilize the code is specified in `tests`. 
```python
  structure_tensor = ImageStructureTensor(img) #The img can be either a string to path or already the pixel matrix
  #Run gradients on the image through a specific operator. The tensor is computed then convoluting the image with a gaussian kernel with a specific simga
  gaussian_sigma=10
  structure_tensor.gradients("Sobel",gaussian_sigma)
  #The structure tensor is computed together with eigenvalues/eigenvectors.
  structure_tensor.eig_2 #Second eigenvalue
  structure_tensor.eig_1 #First eigenvalue
  structure_tensor.E1 #First normalized eigenvector [x,y]
  structure.tensor.E2 #Second normalized eigenvector [x,y]
```
