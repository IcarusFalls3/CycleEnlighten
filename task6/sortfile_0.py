import os

def main():
    directory = "E:\桌面\自己的direct2global\StestB"  # 替换为你的文件目录路径
    images = [f for f in os.listdir(directory) if f.endswith(".png") and f[:-4].isdigit()]
    images.sort(key=lambda x: int(x[:-4]))  # 按数字顺序排序

    new_names = [f"{i}.png" for i in range(len(images))]  # 生成新的文件名列表

    for old_name, new_name in zip(images, new_names):
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")

if __name__ == "__main__":
    main()
