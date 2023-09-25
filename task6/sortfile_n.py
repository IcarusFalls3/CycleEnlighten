import os

folder_path = 'E:\桌面\stest2'  # 替换成你的文件夹路径

# 获取文件夹中的所有文件名
file_names = os.listdir(folder_path)

# 确保文件名是按照数字顺序排列的
file_names.sort(key=lambda x: int(x.split('.')[0]))

# 新的文件名起始数字
new_start_number = 371

for index, old_name in enumerate(file_names):
    old_path = os.path.join(folder_path, old_name)

    # 构造新的文件名
    new_number = new_start_number + index
    new_name = f"{new_number}.png"
    new_path = os.path.join(folder_path, new_name)

    # 如果新的文件名已经存在，则添加后缀来避免重名
    count = 1
    while os.path.exists(new_path):
        new_name = f"{new_number}_{count}.png"
        new_path = os.path.join(folder_path, new_name)
        count += 1

    # 重命名文件
    os.rename(old_path, new_path)

# 单独的循环来删除文件名中的'_1'字符串
for new_name in os.listdir(folder_path):
    if '_1' in new_name:
        new_name_without_underscore = new_name.replace('_1', '')
        new_path = os.path.join(folder_path, new_name)
        new_path_without_underscore = os.path.join(folder_path, new_name_without_underscore)
        os.rename(new_path, new_path_without_underscore)
