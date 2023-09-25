import re
from torch.utils.tensorboard import SummaryWriter

def parse_loss(line):
    # 使用正则表达式从txt文件行中提取损失值
    pattern = r"D_A: (\d+\.\d+) G_A: (\d+\.\d+) cycle_A: (\d+\.\d+) idt_A: (\d+\.\d+) D_B: (\d+\.\d+) G_B: (\d+\.\d+) cycle_B: (\d+\.\d+) idt_B: (\d+\.\d+)"
    match = re.search(pattern, line)
    if match:
        return [float(val) for val in match.groups()]
    else:
        return None

def write_loss_to_event_file(txt_path, event_path):
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()

    # 解析txt文件中的损失值
    loss_values = []
    for line in lines:
        loss = parse_loss(line)
        if loss:
            loss_values.append(loss)

    # 创建一个TensorBoardX的SummaryWriter，并写入事件文件
    writer = SummaryWriter(event_path)
    for step, loss in enumerate(loss_values):
        writer.add_scalar('D_A', loss[0], global_step=step)
        writer.add_scalar('G_A', loss[1], global_step=step)
        writer.add_scalar('cycle_A', loss[2], global_step=step)
        writer.add_scalar('idt_A', loss[3], global_step=step)
        writer.add_scalar('D_B', loss[4], global_step=step)
        writer.add_scalar('G_B', loss[5], global_step=step)
        writer.add_scalar('cycle_B', loss[6], global_step=step)
        writer.add_scalar('idt_B', loss[7], global_step=step)

    writer.close()

if __name__ == "__main__":
    txt_file_path = "./loss_log.txt"  # 替换为您的txt文件路径
    event_file_path = "patch_size/loss_event_patchsize_30"  # 替换为您想要保存event file的路径

    write_loss_to_event_file(txt_file_path, event_file_path)
