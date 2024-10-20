import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.ticker as ticker  # 用于控制刻度格式
from matplotlib import font_manager
from matplotlib.lines import Line2D


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
    ax.set_ylabel('Accumulated return(10^2)', fontsize=12, fontname='Arial')  # y轴文本，字体大小和字体样式
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
    xlabels = [f'{round(x / 10000):1.1f}' for x in xticks]
    xlabels[0] = "0.0"
    ax.set_xticks(xticks, labels=xlabels)
    # 设置y轴刻度
    y_ticks = np.arange(-1000, 1001, 100)
    ylabels = [f'{round(x / 100)}' for x in y_ticks]
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

    # plt.savefig(f'comparison_total_reward2.png')
    # 显示图表
    # plt.show()


def draw_total_ghjg_plt(tag_data):
    """
    绘制规划结果
    :param tag_data:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 5))  # 导出是1000*500像素的效果
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色
    custom_lines = []
    pc_id = 0
    for vehicle_index, vehicle_position_list in tag_data.items():
        color = colors[vehicle_index % len(colors)]  # 循环使用颜色
        x_vals = vehicle_position_list['x']
        y_vals = vehicle_position_list['y']

        # 绘制线条
        if vehicle_index == 0:  # 首辆车是观测车辆
            label = "本算法"
            marker = '>'
        else:
            if vehicle_position_list['x'][0] - tag_data[0]['x'][0] < 20 \
                    and vehicle_position_list['y'][0] == tag_data[0]['y'][0]:
                label = "前车"
            else:
                label = f"旁车{pc_id+1}"
                pc_id+=1
            marker = 'o'

        ax.plot(x_vals, y_vals, label=label, color=color, linewidth=1)

        # 标注每个点
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            if i % 8 == 0:
                ax.scatter(x, y, color=color, s=15, zorder=5, marker=marker)  # 绘制圆点
                ax.text(x + 0.05, y + 0.1, f'{int(i / 8)}', fontsize=10, color=color, ha='center', va='bottom')  # 标注点序号
        custom_lines.append(Line2D([0], [0], color=color, lw=2, marker=marker, label=label))

    ax.set_xlabel('X(m)', fontsize=12, fontname='Times New Roman')  # x轴文本,字体大小和字体样式
    ax.set_ylabel('Y(m)', fontsize=12, fontname='Times New Roman')  # y轴文本，字体大小和字体样式
    ax.grid(True)

    # 设置刻度线样式
    ax.tick_params(axis='y', length=4, direction='in')
    ax.tick_params(axis='x', length=4, direction='in')
    ax.set_xticklabels(ax.get_xticks(), fontsize=10, fontname='Times New Roman')  # x 轴刻度标签字体
    ax.set_yticklabels(ax.get_yticks(), fontsize=10, fontname='Times New Roman')  # y 轴刻度标签字体

    # 设置x轴刻度
    x_min, x_max = ax.get_xlim()
    xticks = np.linspace(x_min, x_max, 6)  # 生成 5 等份的刻度线
    xlabels = [f'{round(x):1.1f}' for x in xticks]
    xlabels[0] = "0.0"
    ax.set_xticks(xticks, labels=xlabels)
    # 设置y轴刻度
    y_ticks = np.arange(0, 16, 4)
    ylabels = [f'{round(x)}' for x in y_ticks]
    ax.set_yticks(y_ticks, labels=ylabels)
    # 初始化图例位置，可以配标题
    legend = ax.legend(handles=custom_lines, loc='lower left', fontsize=12, frameon=True, fancybox=True)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontname('SimSun')  # 设置图例文字的字体
        text.set_fontsize(12)  # 设置图例文字的字体大小
    plt.savefig(f'comparison_total_reward2.png')
    # 显示图表
    plt.show()


def state_ghjg(game_history, time_range):
    state_road_idx = -1
    state_collect = {}
    for state_idx, vehicle_states in enumerate(game_history.observation_history[time_range[0]:time_range[1]]):
        observed_vehicle_y_location = vehicle_states[0, 0, 2]
        if -2 < observed_vehicle_y_location < 2:  # 第0车道
            observed_vehicle_road_idx = 0
        elif 2 < observed_vehicle_y_location < 6:  # 第1车道
            observed_vehicle_road_idx = 1
        elif 6 < observed_vehicle_y_location < 10:  # 第2车道
            observed_vehicle_road_idx = 2
        elif 10 < observed_vehicle_y_location < 14:  # 第3车道
            observed_vehicle_road_idx = 3
        for vehicle_index, vehicle_position in enumerate(vehicle_states[0]):
            if vehicle_position[0] == 1 and -1 < vehicle_index < 6:
                vehicle_pos = vehicle_position[1], vehicle_position[2]
                if not state_collect.get(vehicle_index):
                    state_collect[vehicle_index] = {"x": [vehicle_position[1]],
                                                    "y": [vehicle_position[2]]}
                else:
                    state_collect[vehicle_index]['x'].append(vehicle_position[1])
                    state_collect[vehicle_index]['y'].append(vehicle_position[2])
    draw_total_ghjg_plt(state_collect)


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
    with open(r"C:\Users\lenovo\PycharmProjects\highway_muzero\HighwayEnv-master\test_history2.pkl", "rb") as f:
        game_historys = pickle.load(f)
    # 正常行驶
    state_ghjg(game_historys, [0, 30])
    # 换道时刻
    state_ghjg(game_historys, [30, 80])
