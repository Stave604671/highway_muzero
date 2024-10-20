import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.ticker as ticker  # 用于控制刻度格式


def draw_analyse_plot(logdirs):
    """
    收集tensorboard记录的数据，非有效数据会被自动跳过
    :param logdirs:
    :return:
    """
    pkl_path = r"all_data_tensorflow.pkl"
    if not os.path.exists(pkl_path):
        # 创建一个字典存储每个日志文件的数据
        all_data = {}
        # 遍历每个日志文件并读取数据
        for logdir in logdirs:
            # 加载TensorBoard日志文件
            event_acc = event_accumulator.EventAccumulator(logdir)
            event_acc.Reload()
            # 获取所有的 scalar 标签
            scalar_tags = event_acc.Tags()['scalars']
            # 对每个标签提取step和value数据
            for tag in scalar_tags:
                values = event_acc.Scalars(tag)
                steps = np.array([s.step for s in values])
                data_values = np.array([s.value for s in values])
                # 以标签为键，存储多个日志文件的step和value
                if tag not in all_data:
                    all_data[tag] = []
                all_data[tag].append((steps, data_values, logdir.split('\\')[-3]))  # 存储路径最后一部分以区分来源
        with open(pkl_path, "wb") as f:
            pickle.dump(all_data, f)
    else:
        with open(pkl_path, "rb") as f:
            all_data = pickle.load(f)
    return all_data


def draw_total_reward_plt(tag_data):
    """
    绘制累计reward曲线
    :param tag_data:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 5))  # 导出是1000*500像素的效果
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色
    for i, (steps, data_values, label) in enumerate(tag_data):
        color = colors[i % len(colors)]  # 循环使用颜色
        ax.plot(steps, data_values, label=label, color=color, linewidth=2)

    ax.set_xlabel('Training Steps(10^5)', fontsize=12, fontname='Arial')  # x轴文本,字体大小和字体样式
    ax.set_ylabel('Accumulated return(10^2)', fontsize=12, fontname='Arial')   # y轴文本，字体大小和字体样式
    ax.grid(True)
    ax.legend()

    # 设置刻度线样式
    ax.tick_params(axis='y', length=4, direction='in')
    ax.tick_params(axis='x', length=4, direction='in')
    ax.set_xticklabels(ax.get_xticks(), fontsize=10, fontname='Arial')  # x 轴刻度标签字体
    ax.set_yticklabels(ax.get_yticks(), fontsize=10, fontname='Arial')  # y 轴刻度标签字体

    # 设置x轴刻度
    x_min, x_max = ax.get_xlim()
    xticks = np.linspace(x_min, x_max, 6)  # 生成 5 等份的刻度线
    xlabels = [f'{round(x/10000):1.1f}' for x in xticks]
    xlabels[0] = "0.0"
    ax.set_xticks(xticks, labels=xlabels)
    # 设置y轴刻度
    y_ticks = np.arange(-1000, 1001, 100)
    ylabels = [f'{round(x/100)}' for x in y_ticks]
    ax.set_yticks(y_ticks, labels=ylabels)
    # 初始化图例位置，可以配标题
    legend = ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')  # 设置图例文字的字体
        text.set_fontsize(16)  # 设置图例文字的字体大小

    # 设置图例标题字体
    legend.get_title().set_fontsize(14)  # 设置图例标题的字体大小
    legend.get_title().set_fontname('Monospace')  # 设置图例标题的字体

    plt.savefig(f'comparison_total_reward2.png')
    # 显示图表
    plt.show()


if __name__ == '__main__':
    # 读取tensorflow的结果绘制累计奖励曲线
    root_folder = r"C:\Users\lenovo\PycharmProjects\highway_muzero\highway_sim30"
    tensorboard_log_dir_list = []
    for root, folder, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".pkl") or file_path.endswith(".checkpoint"):
                continue
            else:
                tensorboard_log_dir_list.append(file_path)
    # 调用函数并传入日志文件列表
    out = draw_analyse_plot(tensorboard_log_dir_list)
    draw_total_reward_plt(out['1.Total_reward/1.Total_reward'])
