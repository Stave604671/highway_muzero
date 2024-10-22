import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.ticker as ticker  # 用于控制刻度格式
from matplotlib import font_manager
from matplotlib.lines import Line2D
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
        else:
            observed_vehicle_road_idx = -1
        if state_road_idx != observed_vehicle_road_idx and state_road_idx != -1:
            state_collect.append(state_idx)
        state_road_idx = observed_vehicle_road_idx
    return state_collect


def draw_plot_on_ax(ax, tag_data, choice_flag):
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # 定义一些颜色
    for idx_, (vehicle_index, vehicle_position_list) in enumerate(tag_data.items()):
        color = colors[idx_ % len(colors)]  # 循环使用颜色
        x_vals = range(0, len(vehicle_position_list[choice_flag]))  # x轴是时间
        y_vals = vehicle_position_list[choice_flag]
        cs = CubicSpline(x_vals, y_vals)  # 样条插值
        x_interp = np.linspace(0, len(x_vals)-1, 500)  # 更密集的x轴点
        y_interp = cs(x_interp)  # 使用样条生成平滑的y值
        # 绘制线条
        ax.plot(x_interp, y_interp, label=vehicle_index, color=color, linewidth=1)


def draw_verb_jerk_vy_table(tag_data, xlabel='Time(s)', ylabel="X-speed(m/s)", choice_flag='vx'):
    fig, ax = plt.subplots(figsize=(10, 5))  # 导出是1000*500像素的效果
    draw_plot_on_ax(ax, tag_data, choice_flag)
    # 设置x轴刻度
    x_min, x_max = ax.get_xlim()
    xticks = np.linspace(x_min, x_max, 6)  # 生成 5 等份的刻度线
    xlabels = [f'{round(x):1.1f}' for x in xticks]
    xlabels[0] = "0.0"
    # 设置y轴刻度
    if "v" in choice_flag:
        y_ticks = np.arange(0, 30, 2)
    elif "jerk" in choice_flag:
        y_ticks = np.arange(-160, 160, 20)
    else:
        y_ticks = np.arange(0, 100, 10)
    ylabels = [f'{round(x)}' for x in y_ticks]
    setup_plot_style(ax, xlabel=xlabel, ylabel=ylabel, x_ticks=xticks, xtick_labels=xlabels,
                     y_ticks=y_ticks, ytick_labels=ylabels)
    # 初始化图例位置，可以配标题
    legend = ax.legend( loc='lower left', fontsize=12, frameon=True, fancybox=True)
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontname('SimSun')  # 设置图例文字的字体
        text.set_fontsize(12)  # 设置图例文字的字体大小
    # plt.savefig(f'comparison_total_reward2.png')
    # 显示图表
    plt.show()


def get_vehicle_state_in_game_history(game_history, time_range, vehicle_tag, vehicle_collect_dict):
    for vehicles in game_history.vehicle_history[time_range[0]: time_range[1]]:
        vehicle = vehicles[0]
        vehicle_id = vehicle_tag
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
    return vehicle_collect_dict


if __name__ == '__main__':
    history_pickle_path = r"C:\Users\lenovo\PycharmProjects\highway_muzero\game_history"
    vehicle_collect_dict = {}
    for file in os.listdir(history_pickle_path):
        pickle_file = os.path.join(history_pickle_path, file)
        legend_name = file.replace(".pkl", "").split("_")[2]
        with open(pickle_file, "rb") as f:
            game_history_ = pickle.load(f)
        error_time_range = get_observed_vehicle_road_change(game_history_)
        choice_time_range = [error_time_range[0]-20, error_time_range[0]+20]
        get_vehicle_state_in_game_history(game_history_, choice_time_range, legend_name, vehicle_collect_dict)
    draw_verb_jerk_vy_table(vehicle_collect_dict, ylabel="X-speed(m/s)", choice_flag="vx")
    draw_verb_jerk_vy_table(vehicle_collect_dict, ylabel="X-Jerk(m/s^3)", choice_flag="jerk")
    draw_verb_jerk_vy_table(vehicle_collect_dict, ylabel="Y-speed(m/s)", choice_flag="vy")
        # get_vehicle_state_in_game_history(game_history_, )

    # with open(r"C:\Users\lenovo\PycharmProjects\highway_muzero\HighwayEnv-master\test_history1021.pkl", "rb") as f:
    #     game_historys = pickle.load(f)

    # road_change = get_observed_vehicle_road_change(game_historys)
    # get_vehicle_state_in_game_history(game_historys, time_range=[10, 50])
