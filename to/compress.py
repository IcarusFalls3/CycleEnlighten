from PIL import Image

# 打开原始图像
original_image = Image.open("img.png")

# 将图像调整为256x256分辨率，使用LANCZOS滤波器
compressed_image = original_image.resize((256, 256), Image.LANCZOS)

# 保存压缩后的图像
compressed_image.save("img.png")

# 关闭图像
original_image.close()
compressed_image.close()
