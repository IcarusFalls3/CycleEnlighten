import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.  该类定义了在训练和测试期间使用的选项.

    It also implements several helper functions such as parsing, printing, and saving the options. 它还实现了几个辅助函数，如解析、打印和保存选项。
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class. 它还收集数据集类和模型类中 <modify_commandline_options> 函数定义的其他选项。
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized 重置类；表示类尚未初始化"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test.   定义训练和测试中使用的常用选项。"""
        # basic parameters 基本参数
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        # 有默认值的dataroot
        # parser.add_argument('--dataroot', type=str, default='./datasets/facades', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')


        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters 模型参数
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization | cycle_enlighten]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the last convolution layer 最后一个卷积层中生成器的滤波器数量（滤波器=卷积核） ')
        parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in the first convolution layer 第一卷积层中判别器的滤波器的数量')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')

        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers 仅当判别器使用的是n_layers时使用')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]   实例正常化或批量正常化')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='法线、阶梯和正交的缩放因子.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        # dataset parameters 数据集参数
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly 批处理，true=顺序，false=随机顺序')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data 加载数据的线程')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size 输入批量大小')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size 将图像缩放至此尺寸')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size 然后裁剪成这个尺寸')
        parser.add_argument('--patchSize', type=int, default=64, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help=' 每个数据集允许的最大样本数。如果数据集目录中的数据超过 max_dataset_size，则只加载子集。如果用户没有指定 --max_dataset_size 参数，则将其默认值设置为正无穷大（infinity）')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none] 在加载时对图像进行缩放和裁剪 ')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation 如果指定，则不翻转用于数据增强的图像')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML visdom 和 HTML 的显示窗口大小')
        # additional parameters 附加参数
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        # wandb parameters : wandb能自动记录模型训练过程中的超参数和输出指标,然后可视化和比较结果
        parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
        parser.add_argument('--wandb_project_name', type=str, default='CycleGAN-and-pix2pix', help='specify wandb project name')

        # 局部判别器
        parser.add_argument('--patchD', action='store_true', help='use patch discriminator')
        parser.add_argument('--patchD_3', type=int, default=0, help='choose the number of crop for patch discriminator')
        parser.add_argument('--D_P_times2', action='store_true', help='loss_D_P *= 2')

        self.initialized = True
        return parser

    def gather_options(self):   # 收集选项
        """Initialize our parser with basic options(only once). 使用基本选项初始化我们的解析器（仅一次）。
        Add additional model-specific and dataset-specific options. 添加额外的特定模型和特定数据集选项。
        These options are defined in the <modify_commandline_options> function  这些选项在模型和数据集类中的 <modify_commandline_options> 函数中定义。
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建一个解析对象
            parser = self.initialize(parser)  # 然后向该对象中添加要关注的命令行参数和选项，每一个add_argument方法对应一个要关注的参数或选项；

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):   # 打印选项
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        # k=key（属性名称）,v=value（属性对应的值）
        for k, v in sorted(vars(opt).items()):
            comment = ''  # 在每次循环开始时，初始化一个空字符串 comment
            default = self.parser.get_default(k)  # 获取给定属性名称 k 的默认值
            if v != default:
                comment = '\t[default: %s]' % str(default)
            # {:>25}：这是一个右对齐的占位符，表示将填充值放在 25 个字符的宽度内，并向右对齐。如果填充值长度不足 25 个字符，将在左边用空格进行填充。
            # {:<30}：这是一个左对齐的占位符，表示将填充值放在 30 个字符的宽度内，并向左对齐。如果填充值长度不足 30 个字符，将在右边用空格进行填充。
            # {}：这是一个普通的占位符，用于插入一些可变的内容，比如 comment
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device. 解析我们的选项，创建检查点目录后缀，并设置 gpu 设备。"""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
