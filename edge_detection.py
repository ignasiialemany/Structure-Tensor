import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_img(img):
    return plt.imshow(img)

def load_img(string_img):
    return mpimg.imread('mask.jpg')

def computeConvolution(img,kernel):
    kernel_x_size,kernel_y_size = kernel.shape
    size_x_image,size_y_image = img.shape
    convoluted_matrix = np.zeros([size_x_image-kernel_x_size+1,size_y_image-kernel_y_size+1])

    for i in range (kernel_x_size,size_x_image):
        sliced_rows = img[i-kernel_x_size:i,:]
        for j in range (kernel_y_size,size_y_image):
            temp_matrix = sliced_rows[:,j-kernel_y_size:j]
            convoluted_matrix[i-kernel_x_size,j-kernel_y_size] = np.sum(np.multiply(temp_matrix,kernel))
    return convoluted_matrix

def edgeDetection(img):
    #Based on Sobel matrices
    edge_dx = computeConvolution(img,np.array([[1 ,0 ,-1],[2, 0 ,-2],[1, 0 ,-1]]))
    edge_dy = computeConvolution(img,np.array([[-1 ,-2, -1],[0, 0, 0],[1, 2 ,1]]))
    return edge_dx, edge_dy

def compute_structure_tensor(partial_x,partial_y):
    dx2 = np.multiply(partial_x, partial_x)
    dy2 = np.multiply(partial_y, partial_y)
    dxdy = np.multiply(partial_x, partial_y)
    size_x,size_y = dx2.shape
    structure_tensor = np.zeros([size_x,size_y,2,2])
    eigenvectors = np.zeros([size_x,size_y,2,2])
    eigenvalues = np.zeros([size_x,size_y,2])
    for i in range (0,size_x):
        for j in range (0,size_y):
            structure_tensor[i, j, 0, 0] = dx2[i,j]
            structure_tensor[i, j, 0, 0] = dy2[i,j]
            structure_tensor[i, j, 0, 1], structure_tensor[i, j, 1, 0] = dxdy[i,j], dxdy[i,j]
            eigValues, eigVectors = np.linalg.eig(structure_tensor[i,j,:,:])
            eigenvectors[i,j,:,0] = eigVectors[:,0]
            eigenvectors[i,j,:,1] = eigVectors[:,1]
            eigenvalues[i,j,0] = eigValues[0]
            eigenvalues[i,j,1] = eigValues[1]
    return structure_tensor, eigenvectors, eigenvalues


if __name__ == "__main__":

    img = load_img("mask.jpg")
    partial_x, partial_y = edgeDetection(img)
    structure_tensor, eigenvectors, eigenvalues = compute_structure_tensor(partial_x,partial_y)
    np.save("structure_tensor.npy",structure_tensor)
    np.save("eigenvectors.npy",eigenvectors)
    np.save("eigenvalues.npy",eigenvalues)








