import os

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import numpy as np
from torch.autograd import Variable

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data. 该类实现了 CycleGAN 模型，用于在没有配对数据的情况下学习图像到图像的翻译。

    The model training requires '--dataset_mode unaligned' dataset. 模型训练需要"--dataset_mode unaligned "数据集。
    By default, it uses a '--netG resnet_9blocks' ResNet generator, 默认情况下，它使用"--netG resnet_9blocks"ResNet 生成器、
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix), 一个"--netD basic "判别器（由 pix2pix 引入的 PatchGAN）、
    and a least-square GANs objective ('--gan_mode lsgan'). 以及最小二乘 GAN 损失（'--gan_mode lsgan'）。

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options. 添加新的数据集特定选项，并重写现有选项的默认值。

        Parameters:
            parser 解析器    -- original option parser 原始选项解析器
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options. 是训练阶段还是测试阶段。可以使用此标志添加训练或测试专用选项。

        Returns: 返回
            the modified parser. 修改后的解析器。

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses. 对于 CycleGAN，除了 GAN 损失外，我们还为以下损失引入了 lambda_A、lambda_B 和 lambda_identity。
        A (source domain), B (target domain).  A（源域）、B（目标域）。
        Generators: G_A: A -> B; G_B: B -> A. 生成器： g_a：a -> b；g_b：b -> a。
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A. 判别器： D_A: G_A(A) vs. B; D_B: G_B(B) vs. A。
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper) 前向循环损失：lambda_A * ||G_B(G_A(A)) - A|| （论文中的公式 (2))
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper) 后向循环损失：lambda_B * ||G_A(G_B(B)) - B|| （论文中的公式 (2))
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper) 一致性损失（可选）： lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A)（论文第 5.2 节 "从绘画中生成照片"）。
        Dropout is not used in the original CycleGAN paper.  在最初的 CycleGAN 论文中没有使用 Dropout
        """
        parser.set_defaults(no_dropout=True)  # 默认情况下 CycleGAN 不使用 dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='使用identity映射。将 lambda_identity 设置为 0 以外的值会影响identity映射损失权重的缩放。例如，如果identity损失的权重应比重建损失的权重小 10 倍，请设置 lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        # 用于保存判别器的输出结果张量的计数器
        self.save_count = 0

        """初始化 CycleGAN 类。

         参数:
            opt（选项类）--存储所有实验标志；需要是 BaseOptions的子类
        """
        BaseModel.__init__(self, opt)
        # 指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>。
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # 指定要保存/显示的图像。训练/测试脚本将调用<BaseModel.get_current_visuals>。
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # 如果使用一致性损失，我们还可以直观地看到 idt_B=G_A(B)和 idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # 将 A 和 B 的可视化结合起来
        # 指定要保存到磁盘的模型。训练/测试脚本将调用 <BaseModel.save_networks> 和 <BaseModel.load_networks>。
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # 定义全局判别器
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # itertools.chain函数将两个生成器的参数连接在一起，使得优化器可以同时更新两个生成器的参数
            # betas=(opt.beta1, 0.999)：Adam优化器的两个beta参数，控制梯度的衰减率
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            # 将生成器和判别器的优化器添加到self.optimizers(base_model)列表中。这是因为在训练过程中，我们可能会使用多个优化器进行参数更新
            # （例如，分别更新生成器和判别器的参数），所以将它们放入一个列表中，便于管理和执行参数更新的步骤。
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """从数据加载器中解压缩输入数据，并执行必要的预处理步骤。

        参数：
            input （dict）：包括数据本身及其元数据信息。

        选项 "方向 "可用于交换域 A 和域 B。
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """运行前向传递；由函数 <optimize_parameters> 和 <test> 调用。"""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))


    def save_tensor_to_file(self, tensor, epoch):
        self.save_count += 1

        # 将四维张量转换为字符串，便于保存到文件
        tensor_str = np.array2string(tensor.detach().cpu().numpy(), separator=', ')

        output_directory = 'patchgan_output'  # 已经存在的目录

        # 构建新的文件名和路径
        file_name = os.path.join(output_directory, f'save_tensor_{epoch}.txt')

        # 判断是否已经存在 save_tensor.txt 文件
        try:
            with open(file_name, 'r') as file:
                existing_content = file.read()
        except FileNotFoundError:
            existing_content = ''

        # 使用新的文件名打开文件
        with open(file_name, 'a') as file:
            file.write(f"\n判别结果{self.save_count}\n")
            if existing_content:
                # 如果文件中已经有内容，先写入换行符，然后追加写入四维张量
                file.write('\n')
            file.write(tensor_str)

    def backward_D_basic(self, netD, real, fake, epoch):
        """计算判别器的 GAN 损失

        参数:
            netD（网络）--判别器 D
            real（张量数组）-- 真实图像
            fake（张量数组）-- 由生成器生成的图像

        返回判别器损失。
        我们还会调用 loss_D.backward() 来计算梯度。
        """
        # Real
        pred_real = netD(real)  # 使用判别器netD来对真实图像real进行判别，得到判别器对真实图像的预测结果pred_real。
        # 计算判别器对真实图像的预测结果pred_real与真实标签True之间的GAN损失。
        # 在GAN中，判别器的目标是将真实图像判别为真实的，因此标签为True。
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        # fake通过.detach()方法从计算图中分离出来，这是因为在计算生成器的损失时不需要对生成器的梯度进行更新，只需要更新判别器的梯度。
        # 判别器的目标是将生成图像判别为假的，因此标签为False。
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        """将判别器预测结果输出到本地文件，每一个epoch都输出一遍，顺便在训练的同时，输出5张测试图像"""
        # self.save_tensor_to_file(pred_fake, epoch)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5  # 这里乘以0.5是为了在训练过程中更稳定地更新判别器的参数。
        loss_D.backward()  # 对判别器损失loss_D进行反向传播，计算判别器的参数梯度。
        return loss_D

    def backward_D_A(self, epoch):
        """计算判别器 D_A 的GAN损失"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, epoch)

    def backward_D_B(self, epoch):
        """计算判别器 D_B 的 GAN 损失"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, epoch)


    def backward_G(self):
        """计算生成器 G_A 和 G_B 的损失"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # 计算损失
        # 利用判别器D_A对生成的fake_B进行一个评估，得到一个30*30的patch评估
        # True: 希望判别器 D_A 将生成器 G_A 生成的假图像分类为真实，表示这些假图像应该被判别为真实,即形参 target_is_real = True

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        """计算损失、梯度并更新网络权重；在每次训练迭代中调用"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        # 将判别器（self.netD_A和self.netD_B）的requires_grad属性设置为False，意味着在优化生成器时，不需要计算判别器的梯度，以免影响生成器的参数更新
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # 生成器的反向传播准备步骤，将生成器的梯度（即梯度值）置零，准备接收新的梯度计算
        self.backward_G()  # 生成器的反向传播步骤，计算生成器（G_A和G_B）的梯度，主要涉及生成器的对抗损失和循环一致性损失。这些梯度将用于更新生成器的权重
        self.optimizer_G.step()  # 生成器的参数更新步骤，根据之前计算得到的梯度，使用优化器（Adam等）来更新生成器的权重，使生成器朝着减少损失的方向更新
        # D_A and D_B
        # 将判别器的requires_grad属性设置为True，重新启用判别器的梯度计算
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # 将 D_A 和 D_B 的梯度设为零
        self.backward_D_A(epoch)  # 计算 D_A 的梯度
        self.backward_D_B(epoch)  # 计算 D_B 的梯度
        self.optimizer_D.step()  # 更新 D_A 和 D_B 的权重
