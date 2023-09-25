conv_params = [
    (4, 2, 1),   # 卷积核大小、步长、填充
    (4, 2, 1),
    (4, 2, 1),
    (4, 1, 1),
    (4, 1, 1)
]

receptive_field = 1

for i, (kernel_size, stride, padding) in enumerate(conv_params, start=1):
    receptive_field = (receptive_field - 1) * stride + kernel_size
    print(f"Convolutional Layer {i} Receptive Field: {receptive_field}")
