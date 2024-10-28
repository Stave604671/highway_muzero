import copy
import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.ticker as ticker  # 用于控制刻度格式
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.interpolate import splrep, splev
from scipy.interpolate import CubicSpline


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


def get_observed_vehicle_road_change(game_history):
    """
    根据观测空间获取哪个时刻观测车辆出现碰撞
    :param game_history:
    :return:
    """
    state_road_idx = -1
    state_collect = []
    for state_idx, vehicle_states in enumerate(game_history.vehicle_history):
        observed_vehicle_y_location = vehicle_states[0].position[1]
        if -2 < observed_vehicle_y_location < 2:  # 第0车道
            observed_vehicle_road_idx = 0
        elif 2 < observed_vehicle_y_location < 6:  # 第1车道
            observed_vehicle_road_idx = 1
        elif 6 < observed_vehicle_y_location < 10:  # 第2车道
            observed_vehicle_road_idx = 2
        elif 10 < observed_vehicle_y_location < 14:  # 第3车道
            observed_vehicle_road_idx = 3
        else:
            observed_vehicle_road_idx = -1
        if state_road_idx != observed_vehicle_road_idx and state_road_idx != -1:
            state_collect.append(state_idx)
        state_road_idx = observed_vehicle_road_idx
    return state_collect


def draw_ghpc_table(tag_data, choice_flag='pct', xlabel="X(m)", ylabel="Y(m)"):
    fig, ax = plt.subplots(figsize=(10, 5))  # 导出是1000*500像素的效果
    if choice_flag == "ghpct":
        draw_ghjg_data_on_ax1(ax, tag_data)
    else:
        draw_ghjg_data_pc_on_ax1(ax, tag_data, choice_flag)
    # 设置x轴刻度
    x_min, x_max = ax.get_xlim()
    xticks = np.linspace(x_min, x_max, 10)  # 生成 5 等份的刻度线
    xlabels = [f'{round(x):1.1f}' for x in xticks]
    xlabels[0] = "0.0"
    y_min, y_max = ax.get_ylim()
    y_ticks = np.linspace(y_min, y_max, 5)
    ylabels = [f'{x:.2f}' for x in y_ticks]
    setup_plot_style(ax, xlabel=xlabel, ylabel=ylabel, x_ticks=xticks, xtick_labels=xlabels,
                     y_ticks=y_ticks, ytick_labels=ylabels)
    # 初始化图例位置，可以配标题
    legend = ax.legend( loc='lower left', fontsize=12, frameon=True, fancybox=True)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontname('SimSun')  # 设置图例文字的字体
        text.set_fontsize(12)  # 设置图例文字的字体大小
    plt.savefig(f'{choice_flag}.png')
    # 显示图表





def get_vehicle_state_in_game_history(game_history, time_range, vehicle_tag, vehicle_collect_dict):
    for vehicles in game_history.vehicle_history[time_range[0]: time_range[1]]:
        vehicle = vehicles[0]
        vehicle_id = vehicle_tag

        if not vehicle_collect_dict.get(vehicle_id):
            vehicle_collect_dict[vehicle_id] = {"x": [vehicle.position_recent[0],
                                                      vehicle.position[0]],
                                                "y": [vehicle.position_recent[1],
                                                      vehicle.position[1]],
                                                "new_x": [vehicle.position_recent[0],
                                                          vehicle.new_position[0]],
                                                "new_y": [vehicle.position_recent[1],
                                                          vehicle.new_position[1]],
                                                "vx": [vehicle.get_verb_x],
                                                "vy": [vehicle.get_verb_y],
                                                "jerk_x": [vehicle.get_jerk_x],
                                                "jerk_y": [vehicle.get_jerk_y],
                                                "lateral_deviation": [0, vehicle.position[0] - vehicle.new_position[0]],
                                                # 横向偏差
                                                "longitudinal_deviation": [0,
                                                    vehicle.position[1] - vehicle.new_position[1]]  # 纵向偏差
                                                }

        else:
            vehicle_collect_dict[vehicle_id]['x'].append(vehicle.position[0])
            vehicle_collect_dict[vehicle_id]['y'].append(vehicle.position[1])
            vehicle_collect_dict[vehicle_id]['new_x'].append(vehicle.new_position[0])
            vehicle_collect_dict[vehicle_id]['new_y'].append(vehicle.new_position[1])
            vehicle_collect_dict[vehicle_id]['vx'].append(vehicle.get_verb_x)
            vehicle_collect_dict[vehicle_id]['vy'].append(vehicle.get_verb_y)
            vehicle_collect_dict[vehicle_id]['jerk_x'].append(vehicle.get_jerk_x)
            vehicle_collect_dict[vehicle_id]['jerk_y'].append(vehicle.get_jerk_y)
            vehicle_collect_dict[vehicle_id]['lateral_deviation'].append(vehicle.position[0] - vehicle.new_position[0])
            vehicle_collect_dict[vehicle_id]['longitudinal_deviation'].append(
                vehicle.position[1] - vehicle.new_position[1])

    return vehicle_collect_dict


def draw_ghjg_data_on_ax1(ax, tag_data):
    """
    读取观测空间数据绘制规划结果
    :param ax:
    :param tag_data:
    :return:
    """
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色

    for idx_, (vehicle_index, vehicle_position_list) in enumerate(tag_data.items()):
        color = colors[idx_ % len(colors)]  # 循环使用颜色
        color1 = colors[(idx_ + 1) % len(colors)]

        # 控制路径
        x_vals = vehicle_position_list['x']
        y_vals = vehicle_position_list['y']

        # 使用样条插值
        tck = splrep(x_vals, y_vals, s=0)  # 生成样条对象，s=0 表示没有平滑调整
        x_smooth = np.linspace(min(x_vals), max(x_vals), 200)  # 生成更密集的x坐标
        y_smooth = splev(x_smooth, tck)  # 计算平滑后的y坐标

        ax.plot(x_smooth, y_smooth, label="控制路径", color=color, linewidth=1)

        # 规划路径
        x_vals1 = vehicle_position_list['new_x']
        y_vals1 = vehicle_position_list['new_y']

        # 使用样条插值
        tck1 = splrep(x_vals1, y_vals1, s=0)
        x_smooth1 = np.linspace(min(x_vals1), max(x_vals1), 200)
        y_smooth1 = splev(x_smooth1, tck1)

        ax.plot(x_smooth1, y_smooth1, label="规划路径", color=color1, linewidth=1)


def draw_ghjg_data_pc_on_ax1(ax, tag_data, choice_flag):
    """
    读取观测空间数据绘制规划结果
    :param choice_flag:
    :param ax:
    :param tag_data:
    :return:
    """
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色

    for idx_, (vehicle_index, vehicle_position_list) in enumerate(tag_data.items()):
        color = colors[idx_ % len(colors)]  # 循环使用颜色
        x_vals = range(0, len(vehicle_position_list[choice_flag]))  # x轴是时间

        y_vals = vehicle_position_list[choice_flag]
        label = choice_flag
        cs = CubicSpline(x_vals, y_vals)  # 样条插值
        x_interp = np.linspace(0, len(x_vals)-1, 500)  # 更密集的x轴点
        y_interp = cs(x_interp)  # 使用样条生成平滑的y值
        ax.plot(x_interp, y_interp, label=label, color=color, linewidth=1)


if __name__ == '__main__':
    history_pickle_path = r"C:\Users\lenovo\PycharmProjects\highway_muzero\game_history"
    vehicle_collect_dict = {}
    for file in os.listdir(history_pickle_path):
        pickle_file = os.path.join(history_pickle_path, file)
        legend_name = file.replace(".pkl", "").replace("test_history_", "")
        with open(pickle_file, "rb") as f:
            game_history_ = pickle.load(f)
        error_time_range = get_observed_vehicle_road_change(game_history_)
        choice_time_range = [error_time_range[1]-10, error_time_range[1]+10]
        get_vehicle_state_in_game_history(game_history_, choice_time_range, legend_name, vehicle_collect_dict)

    draw_ghpc_table(vehicle_collect_dict, choice_flag="ghpct")
    draw_ghpc_table(vehicle_collect_dict, choice_flag="lateral_deviation", xlabel="T(s)", ylabel="XError")
    draw_ghpc_table(vehicle_collect_dict, choice_flag="longitudinal_deviation", xlabel="T(s)", ylabel="YError")

        # get_vehicle_state_in_game_history(game_history_, )

    # with open(r"C:\Users\lenovo\PycharmProjects\highway_muzero\HighwayEnv-master\test_history1021.pkl", "rb") as f:
    #     game_historys = pickle.load(f)

    # road_change = get_observed_vehicle_road_change(game_historys)
    # get_vehicle_state_in_game_history(game_historys, time_range=[10, 50])
