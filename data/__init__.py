"""该软件包包含所有与数据加载和预处理相关的模块

 要添加一个名为 "dummy "的自定义数据集类，需要添加一个名为 "dummy_dataset.py "的文件，并定义一个从 BaseDataset 继承而来的子类 "DummyDataset"。
 你需要实现四个函数：
    -- <__init__>：初始化类，首先调用 BaseDataset.__init__(self,opt)。
    -- <__len__>：返回数据集的大小。
    -- <__getitem__>：从数据加载器中获取数据点。
    -- <modify_commandline_options>：（可选）添加特定于数据集的选项和设置默认选项。

现在，你可以通过指定标记"--dataset_mode dummy "来使用数据集类。
更多详情，请参阅我们的模板数据集类 "template_dataset.py"。
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """导入模块 "data/[dataset_name]_dataset.py"。

    在该文件中，名为 DatasetNameDataset() 的类将被实例化。
    类将被实例化。它必须是 BaseDataset 的子类、(......)(......)。
    并且不区分大小写。并且不区分大小写。"""
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)  # 动态导入模块

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    # 遍历 datasetlib 模块（或命名空间）中的所有成员（类、函数等）
    for name, cls in datasetlib.__dict__.items():
        # 检查当前成员的名称（name）是否与 target_dataset_name 相同（忽略大小写）,unaligned_dataset.py里包含UnalignedDataset类
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):  # 检查当前成员（cls）是否是 BaseDataset 类的子类（或者是 BaseDataset 类本身）
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """返回数据集类的静态方法 <modify_commandline_options>."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """根据选项创建数据集。

    该函数封装了 CustomDatasetDataLoader 类。
        这是本软件包与 "train.py"/"test.py "之间的主要接口

    示例
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """数据集类的封装类，可执行多线程数据加载"""

    def __init__(self, opt):
        """初始化该类
        第 1 步：创建一个数据集实例，命名为 [dataset_mode] （数据集模式）
        第 2 步：创建多线程数据加载器。
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)  # find_dataset_using_name返回的是一个类（unaligned_dataset.py里包含UnalignedDataset类）
        self.dataset = dataset_class(opt)  # 直接调用unaligned_dataset里的类的__init__方法
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """返回数据集中的数据个数"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """返回一批数据"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data  # 使用yield语句返回每个批次的数据，而不是一次性将所有数据加载到内存中，这样可以节省内存空间，特别是当数据集很大时
