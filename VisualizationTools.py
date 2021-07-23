import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl


class VisualizationTools:
    def plot_img(self, img):
        plt.imshow(img)
        return plt.show()

    def visualize_eigenvector(self, img, eigenvalue, eigenvector, pixel_res):
        img = img[1:-1:, 1:-1:]
        r, c = np.shape(img)
        gd = pixel_res

        eigV = eigenvector[::gd, ::gd, :]
        eig = eigenvalue[::gd, ::gd]

        fig, ax = plt.subplots(1, 1)
        ax.set_title("Principal Eigenvector")
        ax.imshow(
            img,
            zorder=0,
            alpha=1.0,
            cmap="Greys_r",
            origin="upper",
            interpolation="hermite",
        )
        Y, X = np.mgrid[0:r:gd, 0:c:gd]
        eigV1 = eigV[:, :, 0] * eig[:, :, 0]
        eigV2 = eigV[:, :, 1] * eig[:, :, 0]
        ax.quiver(X, Y, eigV1, eigV2, color="r", minshaft=1, minlength=0)
        fig.savefig("principal_eigenvector.jpg", format="jpg", dpi=1200)
        return plt.show()
