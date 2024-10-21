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
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色
    for i, (steps, data_values, label) in enumerate(tag_data):
        color = colors[i % len(colors)]  # 循环使用颜色
        ax.plot(steps, data_values, label=label, color=color, linewidth=2)
    # 设置x轴刻度
    x_min, x_max = ax.get_xlim()
    xticks = np.linspace(x_min, x_max, 6)  # 生成 5 等份的刻度线
    xlabels = [f'{round(x / 10000):1.1f}' for x in xticks]
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
    # plt.savefig(f'comparison_total_reward2.png')
    # 显示图表
    # plt.show()


def draw_ghjg_data_on_ax1(ax, tag_data):
    """
    读取观测空间数据绘制规划结果
    :param ax:
    :param tag_data:
    :return:
    """
    pc_id = 0
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色
    custom_lines = []

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
                label = f"旁车{pc_id + 1}"
                pc_id += 1
            marker = 'o'
        ax.plot(x_vals, y_vals, label=label, color=color, linewidth=1)
        # 标注每个点
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            if i % 8 == 0:
                ax.scatter(x, y, color=color, s=15, zorder=5, marker=marker)  # 绘制圆点
                ax.text(x + 0.05, y + 0.1, f'{int(i / 8)}', fontsize=10, color=color, ha='center', va='bottom')  # 标注点序号
        custom_lines.append(Line2D([0], [0], color=color, lw=2, marker=marker, label=label))
    return custom_lines


def draw_ghjg_data_on_ax2(ax, tag_data):
    """
    读取车辆状态空间数据绘制规划结果
    :param ax:
    :param tag_data:
    :return:
    """
    pc_id = 0
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色
    custom_lines = []
    observed_vehicle_index = [_ for _ in list(tag_data.keys()) if "_" in _][0]
    for idx_, (vehicle_index, vehicle_position_list) in enumerate(tag_data.items()):
        color = colors[idx_ % len(colors)]  # 循环使用颜色
        x_vals = vehicle_position_list['x']
        y_vals = vehicle_position_list['y']
        # 绘制线条
        if vehicle_index == 0:  # 首辆车是观测车辆
            label = "本算法"
            marker = '>'
        else:
            if x_vals[0] - tag_data[observed_vehicle_index]['x'][0] < 20 \
                    and y_vals[0] == tag_data[observed_vehicle_index]['y'][0]:
                label = "前车"
            else:
                label = f"旁车{pc_id + 1}"
                pc_id += 1
            marker = 'o'
        ax.plot(x_vals, y_vals, label=label, color=color, linewidth=1)
        # 标注每个点
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            if i % 8 == 0:
                ax.scatter(x, y, color=color, s=15, zorder=5, marker=marker)  # 绘制圆点
                ax.text(x + 0.05, y + 0.1, f'{int(i / 8)}', fontsize=10, color=color, ha='center', va='bottom')  # 标注点序号
        custom_lines.append(Line2D([0], [0], color=color, lw=2, marker=marker, label=label))
    return custom_lines


def draw_total_ghjg_plt(tag_data, data_type="observed_space"):
    """
    使用观测空间绘制规划结果{底层按指定规律渲染图片样式，所以修改代码顺序会影响样式}
    :param tag_data:
    :param data_type:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 5))  # 导出是1000*500像素的效果
    if data_type == 'observed_space':
        custom_lines = draw_ghjg_data_on_ax1(ax, tag_data)
    else:
        custom_lines = draw_ghjg_data_on_ax2(ax, tag_data)
    # 设置x轴刻度
    x_min, x_max = ax.get_xlim()
    xticks = np.linspace(x_min, x_max, 6)  # 生成 5 等份的刻度线
    xlabels = [f'{round(x):1.1f}' for x in xticks]
    xlabels[0] = "0.0"
    # 设置y轴刻度
    y_ticks = np.arange(0, 16, 4)
    ylabels = [f'{round(x)}' for x in y_ticks]
    setup_plot_style(ax, xlabel='X(m)', ylabel='Y(m)', x_ticks=xticks, xtick_labels=xlabels,
                     y_ticks=y_ticks, ytick_labels=ylabels)
    # 初始化图例位置，可以配标题
    legend = ax.legend(handles=custom_lines, loc='lower left', fontsize=12, frameon=True, fancybox=True)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontname('SimSun')  # 设置图例文字的字体
        text.set_fontsize(12)  # 设置图例文字的字体大小
    plt.savefig(f'comparison_total_reward2.png')
    # 显示图表
    plt.show()


def draw_ve_plot_on_ax(ax, tag_data):
    pc_id = 0
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色
    custom_lines = []
    observed_vehicle_index = [_ for _ in list(tag_data.keys()) if "_" in _][0]
    for idx_, (vehicle_index, vehicle_position_list) in enumerate(tag_data.items()):
        color = colors[idx_ % len(colors)]  # 循环使用颜色
        x_vals = range(0, len(vehicle_position_list['vx']))
        y_vals = vehicle_position_list['vx']
        # 绘制线条
        if vehicle_index == 0:  # 首辆车是观测车辆
            label = "本算法"
            marker = '>'
        else:
            if x_vals[0] - tag_data[observed_vehicle_index]['x'][0] < 20 \
                    and y_vals[0] == tag_data[observed_vehicle_index]['y'][0]:
                label = "前车"
            else:
                label = f"旁车{pc_id + 1}"
                pc_id += 1
            marker = 'o'
        ax.plot(x_vals, y_vals, label=label, color=color, linewidth=1)


def draw_vx_vy_table(tag_data):
    fig, ax = plt.subplots(figsize=(10, 5))  # 导出是1000*500像素的效果
    custom_lines = draw_ghjg_data_on_ax2(ax, tag_data)
    # 设置x轴刻度
    x_min, x_max = ax.get_xlim()
    xticks = np.linspace(x_min, x_max, 6)  # 生成 5 等份的刻度线
    xlabels = [f'{round(x):1.1f}' for x in xticks]
    xlabels[0] = "0.0"
    # 设置y轴刻度
    y_ticks = np.arange(0, 16, 4)
    ylabels = [f'{round(x)}' for x in y_ticks]
    setup_plot_style(ax, xlabel='X(m)', ylabel='Y(m)', x_ticks=xticks, xtick_labels=xlabels,
                     y_ticks=y_ticks, ytick_labels=ylabels)
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
    """
    从观测空间读取数据并打包成字典绘图
    :param game_history:
    :param time_range:
    :return:
    """
    state_collect = {}
    for state_idx, vehicle_states in enumerate(game_history.observation_history[time_range[0]:time_range[1]]):
        for vehicle_index, vehicle_position in enumerate(vehicle_states[0]):
            if vehicle_position[0] == 1 and -1 < vehicle_index < 6:
                if not state_collect.get(vehicle_index):
                    state_collect[vehicle_index] = {"x": [vehicle_position[1]],
                                                    "y": [vehicle_position[2]]}
                else:
                    state_collect[vehicle_index]['x'].append(vehicle_position[1])
                    state_collect[vehicle_index]['y'].append(vehicle_position[2])
    draw_total_ghjg_plt(state_collect)


def get_observed_vehicle_road_change(game_history):
    """
    根据观测空间获取哪个时刻观测车辆出现碰撞
    :param game_history:
    :return:
    """
    state_road_idx = -1
    state_collect = []
    for state_idx, vehicle_states in enumerate(game_history.observation_history):
        observed_vehicle_y_location = vehicle_states[0, 0, 2]
        if -2 < observed_vehicle_y_location < 2:  # 第0车道
            observed_vehicle_road_idx = 0
        elif 2 < observed_vehicle_y_location < 6:  # 第1车道
            observed_vehicle_road_idx = 1
        elif 6 < observed_vehicle_y_location < 10:  # 第2车道
            observed_vehicle_road_idx = 2
        elif 10 < observed_vehicle_y_location < 14:  # 第3车道
            observed_vehicle_road_idx = 3
        if state_road_idx != observed_vehicle_road_idx and state_road_idx != -1:
            state_collect.append(state_idx)
        state_road_idx = observed_vehicle_road_idx
    return state_collect


def read_folder_get_tensorboard_result():
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


def get_vehicle_state_in_game_history(game_history, time_range):
    vehicle_collect_dict = {}
    for vehicles in game_history.vehicle_history[time_range[0]: time_range[1]]:
        for vehicle_idx, vehicle in enumerate(vehicles[:5]):
            vehicle_id = str(vehicle_idx)
            if vehicle.is_observed:
                vehicle_id = f"{vehicle_id}_obs"
            if not vehicle_collect_dict.get(vehicle_id):
                vehicle_collect_dict[vehicle_id] = {"x": [vehicle.position[0]],
                                                    "y": [vehicle.position[1]],
                                                    "vx": [vehicle.get_verb_x],
                                                    "vy": [vehicle.get_verb_y],
                                                    "jerk": [vehicle.get_jerk]}
            else:
                vehicle_collect_dict[vehicle_id]['x'].append(vehicle.position[0])
                vehicle_collect_dict[vehicle_id]['y'].append(vehicle.position[1])
                vehicle_collect_dict[vehicle_id]['vx'].append(vehicle.get_verb_x)
                vehicle_collect_dict[vehicle_id]['vy'].append(vehicle.get_verb_y)
                vehicle_collect_dict[vehicle_id]['jerk'].append(vehicle.get_jerk)
    draw_total_ghjg_plt(vehicle_collect_dict, data_type="vehicle_space")


if __name__ == '__main__':
    # read_folder_get_tensorboard_result()
    with open(r"C:\Users\lenovo\PycharmProjects\highway_muzero\HighwayEnv-master\test_history1021.pkl", "rb") as f:
        game_historys = pickle.load(f)
    road_change = get_observed_vehicle_road_change(game_historys)
    get_vehicle_state_in_game_history(game_historys, time_range=[10, 50])
    # 正常行驶
    # state_ghjg(game_historys, [0, 30])
    # 换道时刻
    # state_ghjg(game_historys, [30, 80])
