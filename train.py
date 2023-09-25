"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':         # 如果这个脚本作为主脚本使用，那么就运行下方的东西
    opt = TrainOptions().parse()   # 把TrainOption实例化成对象，然后用parse进行解析，这样形成一个结果，赋给opt，也就是说，opt是解析出来的结果
    dataset = create_dataset(opt)  # 根据这个结果去创建数据集
    dataset_size = len(dataset)    # 获取数据集中样本的数量
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # 创建模型
    model.setup(opt)               # 根据opt创建合适的学习率调整策略、导入网络并打印
    visualizer = Visualizer(opt)   # 根据opt创建可视化实例
    total_iters = 0                # 训练迭代次数

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # 迭代过程的开始，opt.epoch_count是从哪个epoch开始，opt.n_epochs是以初始lr训练100个epoch，opt.n_epochs_decay是lr衰退的100个epoch
        epoch_start_time = time.time()  # 获取这个epoch开始的时间
        iter_data_time = time.time()    # 本轮epoch导入数据的时间
        epoch_iter = 0                  # 在本轮epoch当中的第几次迭代
        visualizer.reset()              # 可视化机器的重置,保证在每个epoch里它至少有一次保存图片
        model.update_learning_rate()    # 在每次epoch之前率先更新一下学习率

        # output_directory = 'patchgan_output'  # 已经存在的目录
        #
        # # 构建新的文件名和路径
        # file_name = os.path.join(output_directory, f'save_tensor_{epoch}.txt')
        #
        # # 使用新的文件名打开文件
        # with open(file_name, 'a') as file:
        #     file.write(f"\nepoch{epoch}\n")

        for i, data in enumerate(dataset):  # 每个epoch内部的循环,enumerate函数的作用是同时列出数据和下标,i是batch的编号，data不是一张图，而是一个batch的图
            iter_start_time = time.time()  # 本次iteration开始的时间
            if total_iters % opt.print_freq == 0:  # 如果总迭代次数total_iters到了opt.print_freq的整倍数
                t_data = iter_start_time - iter_data_time  # 计算t_data，也就是本轮iteration开始的时刻到本轮epoch导入数据的时间已经过去了多久

            total_iters += opt.batch_size  # 一共多少个数据参与了训练
            epoch_iter += opt.batch_size  # 本轮epoch里有多少数据参与了迭代
            model.set_input(data)         # 把每一个数据解包
            model.optimize_parameters()   # 参数优化,这些都是在models当中的base_model.py当中定义的


            if total_iters % opt.display_freq == 0:   # 如果本轮epoch当中已经参与迭代的样本总数是opt.display_freq的整数倍
                save_result = total_iters % opt.update_html_freq == 0  # 返回一个叫save_result的布尔值，用于判定是否需要存出结果到html文件里
                model.compute_visuals()  # 只有在着色任务中才有用，是展示图片的命令，其他的模型中compute_visuals函数只有一个命令，那就是pass
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)  # 存储到html文件里的命令，其中save_result就决定本行是否执行，可以参见util文件夹里的visualizer.py

            if total_iters % opt.print_freq == 0:    # 如果本轮epoch当中已经参与迭代的样本总数是opt.print_freq的整数倍
                losses = model.get_current_losses()  # 获取当前的损失函数
                t_comp = (time.time() - iter_start_time) / opt.batch_size  # 计算每个图片所用的时间
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)  # 输出当前的损失值
                if opt.display_id > 0:  # 如果window id of the web display这个值大于0
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)  # 那么就利用plot_current_losses函数输出，参数的含义可以点击util文件夹下的visuallizer.py去查看相应的函数。

            if total_iters % opt.save_latest_freq == 0:   # 如果本轮epoch当中已经参与迭代的样本总数是opt.save_latest_freq的整数倍
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))  #
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'  # 设置保存后缀
                model.save_networks(save_suffix)  # 保存模型

            iter_data_time = time.time()  # 重新获取时间
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))  #
            model.save_networks('latest')  # 保存模型的名称
            model.save_networks(epoch)  # 保存模型

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))  #
