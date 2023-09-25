import random
import torch


class ImagePool():
    """该类实现了一个图像缓冲区，用于存储之前生成的图像。
    通过该缓冲区，我们可以使用历史生成的图像而不是最新生成的图像来更新判别器
    而不是最新生成的图像来更新判别器。
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """从图片池中返回一张图片。

        参数：
            images：生成器最新生成的图像

        从缓冲区返回图像。

        在 50/100 时，缓冲区将返回输入的图像。
        50/100 时，缓冲区将返回之前存储在缓冲区中的图像、
        并将当前图像插入缓冲区。"""
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
