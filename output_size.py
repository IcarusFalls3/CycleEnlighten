import torch
import torch.nn as nn

# 定义输入图像尺寸和通道数
batch_size = 1
channels = 3
height = 256
width = 256

# 定义卷积核大小、步长和填充
kernel_size = 4
stride = 2
padding = 1

kernel_size = 4

# 定义五层卷积层
conv_layers = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size, 2, 1),
    nn.Conv2d(64, 128, kernel_size, 2, 1),
    nn.Conv2d(128, 256, kernel_size, 2, 1),
    nn.Conv2d(256, 512, kernel_size, 1, 1),
    nn.Conv2d(512, 1, kernel_size, 1, 1)
)

# 计算每一层的输出尺寸和感受野大小
input_shape = (batch_size, channels, height, width)
output_shapes = []
receptive_field = 1

for i, conv_layer in enumerate(conv_layers, start=1):
    output_shape = conv_layer(torch.zeros(input_shape)).shape
    output_shapes.append(output_shape)

    # 计算感受野大小
    receptive_field = (receptive_field - 1) * conv_layer.stride[0] + conv_layer.kernel_size[0]

    print(f"Convolutional Layer {i} Output Shape: {output_shape} | Receptive Field: {receptive_field}")
    input_shape = output_shape