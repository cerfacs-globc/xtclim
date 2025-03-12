import imageio
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

to_pil_image = transforms.ToPILImage()


def image_to_vid(images, output_path: str = "./outputs"):
    # save evolving images along the learning and get the video
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(output_path + "/generated_images.gif", imgs)


def save_reconstructed_images(recon_images, epoch, season="", output_path: str = "./outputs"):
    # save all reconstructed images at each epoch
    save_image(recon_images.cpu(), output_path + f"/image_record/{season}output{epoch}.jpg")


def save_ex(recon_ex, epoch, season="", output_path: str = "./outputs"):
    # save an example of image at a given epoch
    save_image(recon_ex.cpu(), output_path + f"/image_record/{season}ex{epoch}.jpg")


def save_loss_plot(train_loss, valid_loss, season="", output_path: str = "./outputs"):
    # saves the plot of both losses evolutions
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(valid_loss, color="red", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path + f"/{season}loss.jpg")
    plt.show()
