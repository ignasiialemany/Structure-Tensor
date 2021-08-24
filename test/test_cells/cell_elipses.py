from ImageStructureTensor import ImageStructureTensor
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import imageio
import numpy as np
import math
from skimage import io
import scipy.signal as spy
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_closing
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from ImageStructureTensor import ImageStructureTensor
from PIL import Image


def plot_largest_eigenvalue():
    # Read image
    im = io.imread("Synth2D.tif")
    im = np.array(im, dtype=np.float64)
    im = im / np.max(im)
    structure_tensor = ImageStructureTensor(im)
    #plot_convergence(structure_tensor)
    plot_results(structure_tensor,10)

def plot_convergence(structure_tensor):
    ratio_axis = []
    sigma_values=[]
    for sigma in range (10,14,1):
        structure_tensor.gradients("Zhang2014",sigma)
        sigma_values.append(sigma)
        ratio_axis.append(np.sqrt(np.max(structure_tensor.eig_2)/np.max(structure_tensor.eig_1)))

    plt.axhline(y=0.5, color='b', linestyle='--')
    plt.plot(sigma_values,ratio_axis,"x--k")
    plt.xlabel("$\sigma$ (Gaussian)")
    plt.ylabel("$\sqrt{\lambda_2 / \lambda_1}$")
    plt.savefig("convergence_ratio_eigenvalues.png",dpi=800)
    plt.show()

#Create single ellipse
def create_single_ellipse(image,x,y,x0,y0):
    #theta=np.deg2rad(0)
    theta = np.random.rand()*2*np.pi*1.3
    U=[[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]
    diag_matrix=[[1/8,0],[0,1]]
    A = np.transpose(U).dot(diag_matrix).dot(U)
    xy_vector = np.hstack([x.ravel()-x0]) * np.hstack([y.ravel()-y0])
    el = (A[0, 0]) * np.hstack([x.ravel()-x0]) ** 2 + (A[1, 1]) * np.hstack([y.ravel()-y0]) ** 2 + (A[1, 0])*xy_vector + A[0,1]*xy_vector
    #el = np.transpose(positions).dot(A).dot(positions)
    index = np.abs(el)>=1
    el = np.exp((-1)/(1-np.abs(el)**2))
    el[index]=0
    image = image + np.reshape(el,([len(x),len(y)]))
    return image

def create_map_of_ellipses():
    #All grid
    x_grid, y_grid = np.meshgrid(np.arange(0, 40, 0.09), np.arange(0, 40, 0.09))
    #Collection of ellipses
    x_ellipses,y_ellipses = np.meshgrid(np.arange(4,38,8),np.arange(4,38,8))
    image = np.zeros([len(x_grid),len(y_grid)])
    for x,y in zip(x_ellipses.ravel(),y_ellipses.ravel()):
       image = create_single_ellipse(image,x_grid,y_grid,x,y)

    fig = plt.figure()
    image/=np.max(image)
    image *= 255
    image = np.uint8(image)
    png2 = Image.fromarray(image)
    # Save as TIFF
    png2.save("theta_vardeg2.tif")
    png2.close()

def plot_results(structure_tensor,sigma):

    structure_tensor.gradients("Zhang2014", sigma)
    thres = threshold_otsu(structure_tensor.img)
    binary = np.where(structure_tensor.img > thres, 255, 0)
    labeled_image, num_features = label(binary)
    properties = regionprops(labeled_image)
    centroids = np.array([list(properties[x].centroid) for x in range(0, len(properties))], dtype=np.int64)

    x, y = np.meshgrid(np.linspace(0,structure_tensor.img_size_x-5,15),np.linspace(0,structure_tensor.img_size_y-5,15))
    x = np.array([int(element) for element in x.ravel()])
    y = np.array([int(element) for element in y.ravel()])

    x, y = np.meshgrid(np.arange(0,structure_tensor.img_size_x,20),np.arange(0,structure_tensor.img_size_y,20))
    a1 = np.transpose(structure_tensor.E1[:,:,1])
    a2 = np.transpose(structure_tensor.E1[:,:,0])

    b1 = np.transpose(structure_tensor.E2[:, :, 1])
    b2 = np.transpose(structure_tensor.E2[:, :, 0])

    plt.figure()
    plt.imshow(structure_tensor.img, origin="lower")
    plt.quiver(x, y, a1[x,y],
               a2[x,y], headaxislength=0, headlength=0, color="red")
    plt.title("$E_1$ direction")
    plt.savefig("e1_direction_0theta.png",dpi=800)
    plt.show()

    plt.figure()
    plt.imshow(structure_tensor.img,origin='lower')
    plt.quiver(x, y, b1[x, y],
               b2[x, y], headaxislength=0, headlength=0, color="red")
    plt.title("$E_2$ direction")
    plt.savefig("e2_direction_0theta.png", dpi=800)
    plt.show()

    plt.figure()
    b = plt.imshow(structure_tensor.eig_1)
    plt.colorbar(b)
    plt.title("$\lambda_1$")
    plt.savefig("lambda1_0theta.png", dpi=800)
    plt.show()

    plt.figure()
    b = plt.imshow(structure_tensor.eig_2)
    plt.colorbar(b)
    plt.title("$\lambda_2$")
    plt.savefig("lambda2_0theta.png", dpi=800)
    plt.show()

#create_map_of_ellipses()
plot_largest_eigenvalue()