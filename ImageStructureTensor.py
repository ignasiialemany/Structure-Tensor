import matplotlib.image as mpimg
import numpy as np
import pickle
import scipy.signal as spy
import matplotlib.pyplot as plt


class ImageStructureTensor:
    def __init__(self, img_string):
        self.loadImage(img_string)

    def loadImage(self, img_string):
        if type(img_string) is str:
            self.img = mpimg.imread(img_string)
        else:
            self.img = img_string
        self.img_size_x, self.img_size_y = self.img.shape

    def dumpInFile(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    def computeConvolution(self, matrix, kernel):
        return np.array(spy.convolve2d(matrix, kernel, boundary="symm", mode="same"))

    def gradients(self, operator, sigma):
        if operator == "Sobel":
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        elif operator == "Normal_Gradient":
            kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        elif operator == "Zhang2014":
            kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) * 1 / 32
            kernel_y = np.transpose(kernel_x)
        else:
            raise NameError("This operator has not been implemented yet")

        self.gradient_x = self.computeConvolution(self.img, kernel_x)
        self.gradient_y = self.computeConvolution(self.img, kernel_y)

        self.gradient_x2 = np.multiply(self.gradient_x, self.gradient_x)
        self.gradient_y2 = np.multiply(self.gradient_y, self.gradient_y)
        self.gradient_xy = np.multiply(self.gradient_x, self.gradient_y)

        # Gaussian filters
        g_kern = self.gkern(self.img_size_x, self.img_size_y, sigma)
        g_kern = np.fft.fftshift(g_kern)
        fft_gkern = np.fft.fft2(g_kern)
        fft_gradientx2 = np.fft.fft2(self.gradient_x2)
        fft_gradienty2 = np.fft.fft2(self.gradient_y2)
        fft_gradientxy = np.fft.fft2(self.gradient_xy)

        self.init_tensor()
        self.tensor[:, :, 0, 0] = np.fft.ifft2(fft_gradientx2 * fft_gkern)
        self.tensor[:, :, 0, 1] = np.fft.ifft2(fft_gradientxy * fft_gkern)
        self.tensor[:, :, 1, 0] = self.tensor[:, :, 0, 1]
        self.tensor[:, :, 1, 1] = np.fft.ifft2(fft_gradienty2 * fft_gkern)
        self.computeEig()

    def init_tensor(self):
        self.tensor = np.zeros(
            [self.gradient_x2.shape[0], self.gradient_x2.shape[1], 2, 2]
        )

    def computeEig(self):
        trace = self.tensor[:, :, 0, 0] + self.tensor[:, :, 1, 1]
        det = (self.tensor[:, :, 0, 0] * self.tensor[:, :, 1, 1]) - (
            self.tensor[:, :, 0, 1] * self.tensor[:, :, 1, 0]
        )
        DETA = np.sqrt(trace ** 2 - 4 * det)
        # https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
        # https://croninprojects.org/Vince/Geodesy/FindingEigenvectors.pdf

        self.eig_2 = (trace - DETA) / 2
        self.E2 = np.zeros([len(self.eig_2), len(self.eig_2[0]), 2])
        self.E2[:, :, 0] = -self.tensor[:, :, 0, 1]
        self.E2[:, :, 1] = self.tensor[:, :, 1, 1] - self.eig_2

        self.eig_1 = (trace + DETA) / 2
        self.E1 = np.zeros([len(self.eig_1), len(self.eig_1[0]), 2])
        self.E1[:, :, 0] = self.tensor[:, :, 0, 0] - self.eig_1
        self.E1[:, :, 1] = -self.tensor[:, :, 0, 1]

        norm_E1 = np.linalg.norm(self.E1[:, :], axis=2)
        norm_E2 = np.linalg.norm(self.E2[:, :], axis=2)
        self.E1[:, :, 0] /= norm_E1
        self.E1[:, :, 1] /= norm_E1
        self.E2[:, :, 0] /= norm_E2
        self.E2[:, :, 1] /= norm_E2

    def gkern(self, nx, ny, sig):
        ax = np.linspace(-(nx - 1) / 2.0, (nx - 1) / 2.0, nx)
        ay = np.linspace(-(ny - 1) / 2.0, (ny - 1) / 2.0, ny)
        xx, yy = np.meshgrid(ax, ay)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        return kernel / np.sum(kernel)
