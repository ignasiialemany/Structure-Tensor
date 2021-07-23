import matplotlib.image as mpimg
import numpy as np
import pickle


class ImageStructureTensor:
    def __init__(self, img_string, operator):
        self.loadImage(img_string)
        self.edgeDetection(operator)
        self.init_tensor()

    def loadImage(self, img_string):
        self.img = mpimg.imread(img_string)
        self.img_size_x, self.img_size_y = self.img.shape

    def dumpInFile(self, file):
        with open("temp//" + file, "wb") as f:
            pickle.dump(self, f)

    def computeConvolution(self, img, kernel):
        convoluted_matrix = np.zeros(
            [
                self.img_size_x - self.kernel_size + 1,
                self.img_size_y - self.kernel_size + 1,
            ]
        )
        for i in range(self.kernel_size, self.img_size_x):
            sliced_rows = img[i - self.kernel_size : i, :]
            for j in range(self.kernel_size, self.img_size_y):
                temp_matrix = sliced_rows[:, j - self.kernel_size : j]
                convoluted_matrix[i - self.kernel_size, j - self.kernel_size] = np.sum(
                    np.multiply(temp_matrix, kernel)
                )
        return convoluted_matrix

    def edgeDetection(self, operator):
        if operator == "Sobel":
            self.kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            self.kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            self.kernel_size = self.kernel_x.shape[0]
        else:
            raise NameError("This operator has not been implemented yet")

        self.gradient_x = self.computeConvolution(self.img, self.kernel_x)
        self.gradient_y = self.computeConvolution(self.img, self.kernel_y)
        self.gradient_x2 = np.multiply(self.gradient_x, self.gradient_x)
        self.gradient_y2 = np.multiply(self.gradient_y, self.gradient_y)
        self.gradient_xy = np.multiply(self.gradient_y, self.gradient_x)

    def init_tensor(self):
        self.tensor = np.zeros(
            [self.gradient_x.shape[0], self.gradient_x.shape[1], 2, 2]
        )
        self.eigVectors = np.zeros(
            [self.gradient_x.shape[0], self.gradient_x.shape[1], 2, 2]
        )
        self.eigValues = np.zeros(
            [self.gradient_x.shape[0], self.gradient_x.shape[1], 2]
        )

    def computeStructureTensor(self):
        for i in range(0, self.tensor.shape[0]):
            for j in range(0, self.tensor.shape[1]):
                self.tensor[i, j, :, :] = [
                    [self.gradient_x2[i, j], self.gradient_xy[i, j]],
                    [self.gradient_xy[i, j], self.gradient_y2[i, j]],
                ]
                eigValues, eigVectors = np.linalg.eig(self.tensor[i, j, :, :])
                indices = np.argsort(eigValues)[::-1]
                self.eigVectors[i, j, :, 0] = eigVectors[:, indices[0]]
                self.eigVectors[i, j, :, 1] = eigVectors[:, indices[1]]
                self.eigValues[i, j, 0] = eigValues[indices[0]]
                self.eigValues[i, j, 1] = eigValues[indices[1]]
