import numpy as np
import matplotlib.pyplot as plt

# 读取文本文件中的数据
with open("a_test_tensor.txt", "r") as file:
    data_str = file.read()

# 将数据字符串转换为numpy数组
data_array = np.array(eval(data_str))

# 将数组中的数据进行reshape
reshaped_data = np.reshape(data_array, (1, 1, 30, 30))

# 转换为灰度图像
gray_image = np.squeeze(reshaped_data)  # 移除单维度
plt.imshow(gray_image, cmap="gray")  # 使用灰度色彩映射
plt.axis("off")  # 关闭坐标轴
plt.show()

# 保存图像到本地
plt.imsave("gray_image_1_219.png", gray_image, cmap="gray")
