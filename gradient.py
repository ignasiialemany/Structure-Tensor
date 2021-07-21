import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#img = mpimg.imread('mask.jpg')
#imgplot = plt.imshow(img)


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
    #Based on Sobel
    edge_dx = computeConvolution(img,np.array([[1 ,0 ,-1],[2, 0 ,-2],[1, 0 ,-1]]))
    edge_dy = computeConvolution(img,np.array([[-1 ,-2, -1],[0, 0, 0],[1, 2 ,1]]))
    return edge_dx, edge_dy

if __name__ == "__main__":

    img = load_img("mask.jpg")
    partial_x, partial_y = edgeDetection(img)





