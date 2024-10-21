import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def process_tensorboard_cache(logdirs):
    """
    收集tensorboard记录的数据并生成缓存，非有效数据会被自动跳过
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


def setup_plot_style(ax,
                     xlabel='X(m)',
                     ylabel='Y(m)',
                     x_ticks=None,
                     y_ticks=None,
                     xtick_labels=None,
                     ytick_labels=None):
    """
    设置图表样式，包括刻度线和x、y轴文本
    :param ax: matplotlib Axes 对象
    :param xlabel: x轴文本
    :param ylabel: y轴文本
    :param x_ticks: x轴刻度位置
    :param y_ticks: y轴刻度位置
    :param xtick_labels: x轴刻度标签
    :param ytick_labels: y轴刻度标签
    """
    ax.set_xlabel(xlabel, fontsize=12, fontname='Times New Roman')
    ax.set_ylabel(ylabel, fontsize=12, fontname='Times New Roman')
    ax.grid(True)

    # 设置刻度线样式
    ax.tick_params(axis='y', length=4, direction='in')
    ax.tick_params(axis='x', length=4, direction='in')

    # 设置x轴和y轴刻度
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        if xtick_labels is not None:
            ax.set_xticklabels(xtick_labels, fontsize=10, fontname='Times New Roman')
        else:
            ax.set_xticklabels(x_ticks, fontsize=10, fontname='Times New Roman')

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        if ytick_labels is not None:
            ax.set_yticklabels(ytick_labels, fontsize=10, fontname='Times New Roman')
        else:
            ax.set_yticklabels(y_ticks, fontsize=10, fontname='Times New Roman')


def draw_total_reward_plt(tag_data):
    """
    绘制累计reward曲线
    :param tag_data:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 5))  # 导出是1000*500像素的效果
    colors = ['darkblue', 'olive', 'firebrick', 'gold', 'mediumpurple']  # 定义一些颜色
    for i, (steps, data_values, label) in enumerate(tag_data):
        color = colors[i % len(colors)]  # 循环使用颜色
        ax.plot(steps, data_values, label=label, color=color, linewidth=1)
    # 设置x轴刻度
    xticks = np.arange(0, 80000, 4000)  # 生成 5 等份的刻度线
    xlabels = [f'{x / 10000:1.1f}' for x in xticks]
    xlabels[0] = "0.0"
    # 设置y轴刻度
    y_ticks = np.arange(-1000, 1001, 100)
    ylabels = [f'{round(x / 100)}' for x in y_ticks]
    setup_plot_style(ax, xlabel='Training Steps(10^5)', ylabel='Accumulated return(10^2)',
                     x_ticks=xticks, xtick_labels=xlabels, y_ticks=y_ticks, ytick_labels=ylabels)
    # 初始化图例位置，可以配标题
    legend = ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')  # 设置图例文字的字体
        text.set_fontsize(16)  # 设置图例文字的字体大小
    # 设置图例标题字体
    legend.get_title().set_fontsize(14)  # 设置图例标题的字体大小
    legend.get_title().set_fontname('Times New Roman')  # 设置图例标题的字体
    plt.savefig(f'comparison_total_reward2.png')
    # 显示图表
    plt.show()


def read_folder_get_tensorboard_result():
    # 读取tensorflow的结果绘制累计奖励曲线
    root_folder = r"C:\Users\lenovo\PycharmProjects\highway_muzero\plot_img"
    tensorboard_log_dir_list = []
    for root, folder, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".pkl") or file_path.endswith(".checkpoint"):
                continue
            else:
                tensorboard_log_dir_list.append(file_path)
    # 调用函数并传入日志文件列表
    tensorboard_cache = process_tensorboard_cache(tensorboard_log_dir_list)
    draw_total_reward_plt(tensorboard_cache['1.Total_reward/1.Total_reward'])


if __name__ == '__main__':
    read_folder_get_tensorboard_result()
