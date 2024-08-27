import datetime
import pathlib
import time

import gymnasium as gym
import numpy
import torch
import numpy as np
from ray import logger
from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0  # 随机数种子,用于固定随机性方便复现
        self.max_num_gpus = None  # 固定使用gpu的最大数量.使用单个gpu会更快,没有配置的话会默认使用所有gpu

        # Game
        self.observation_shape = (1, 21, 7)  # 游戏观测空间的维度,如果观测空间三维无所谓,如果是一维,需要配成(1,1,x)
        self.action_space = 2  # 动作空间的大小
        self.players = [i for i in range(1)]  # 玩家的数量,车辆换道场景观测和控制车辆只有一个,为1就行
        self.stacked_observations = 0  # 观测时叠加的历史观察数量（包括过去的动作）。

        # Evaluate
        self.muzero_player = 0  # 用于区分多玩家环境中谁是 MuZero 控制的玩家。自我对弈默认是0.
        self.opponent = None  # MuZero 面对的对手，用于评估在多人游戏中的进展。可以是 "random" 或 "expert"，如果在游戏类中实现了对手

        # Self-Play
        self.num_workers = 2  # 定义了同时进行 Self-Play 的工作线程数量，这些线程负责生成训练样本并将其存储到回放缓冲区中。
        self.selfplay_on_gpu = False  # 是否在gpu进行自我博弈,打开后速度变快,但是显存开支会高很多
        self.max_moves = 1000  # 每场游戏的最大游戏次数,未发生碰撞,或者没有达到这个次数,单场游戏都不停止
        self.num_simulations = 35  # 执行指定次数的模拟，每次模拟从根节点开始进行搜索和更新,
        """
        discount 参数对 Total Reward 曲线的影响可以从以下几个方面来理解：
        (1)未来回报的重要性：
        较小的 discount：如果 discount 值较小，模型会更关注短期回报，导致它在每一步的策略中更倾向于最大化当前的奖励。这种情况下，Total Reward 曲线可能会快速上升，但也容易达到一个平台期，因为模型忽略了远期回报的影响，容易陷入局部最优。
        较大的 discount：如果 discount 值较大，模型会同时考虑短期和长期回报。这通常会让Total Reward曲线在前期上升较慢，因为模型更注重长期规划，愿意在短期内做出一些“牺牲”。但是，随着训练的进行，Total Reward 曲线可能会更平滑地上升，最终达到更高的总回报。
        (2)曲线的波动性：
        较小的 discount：由于关注短期回报，Total Reward 曲线可能会更加波动，因为模型会快速调整策略以适应即时回报的变化。这可能导致曲线在某些时候突然上升或下降，缺乏稳定性。
        较大的 discount：考虑到长远的回报，模型的决策会更加稳定，Total Reward 曲线也会变得相对平滑，波动性减小。虽然曲线的增长速度在前期可能较慢，但随着时间的推移，它通常能持续上升。
        (3)长期收益的体现：
        较小的 discount：在总训练步数较长的情况下，Total Reward 可能会在早期快速上升，然后趋于平稳甚至停滞，因为模型已经优化了短期回报，而未能充分利用长期回报。
        较大的 discount：Total Reward 的上升可能是逐步且持久的，因为模型能够逐步发现并利用长期的策略，最终获得更高的总回报。
        总结：discount 值的选择会影响 Total Reward 曲线的上升速度、平滑度和最终的总回报。一般情况下，较大的 discount 值能带来更稳定、更长期的回报，Total Reward 曲线更平滑且在后期继续上升。较小的 discount 值则可能带来更快的初期收益，但容易波动，并且总回报可能较低。
        """
        self.discount = 0.985  # 长期回报的折扣因子
        self.temperature_threshold = 800  # 单次play_games的温度阈值,当前的play_games内,最大移动self.max_moves次,moves的次数超过这个阈值后,温度直接为0,低于这个次数时,启用visit_softmax_temperature_fn获取温度数值
        # 'uniform' or 'density'
        # 在自动驾驶换道场景下：如果你希望模型重点考虑某些特定的换道策略（比如避免某些危险的换道动作），选择 density。
        # 如果你希望模型自行探索各种可能的换道策略，选择 uniform。
        self.node_prior = 'uniform'

        # UCB formula
        self.pb_c_base = 19652  # 数值越大,更倾向于利用选择已知效果较好的动作,而非探索新动作
        self.pb_c_init = 1.2  # 初始化参数,对探索奖励有一个固定的提升作用.数值越大,初期的探索越多.反之更依赖已知动作

        # Progressive widening parameter
        # pw_alpha用来调节何时对节点进行渐进扩展。渐进扩展的基本思想是，当一个节点的访问次数较少时，增加它的子节点的数量以增加探索的多样性，
        # 从而避免过早地确定子节点的评估结果。
        self.pw_alpha = 0.4

        # network_config2
        self.network = "resnet"
        # Residual Network
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 256  # Number of channels in the ResNet
        # Define channels for each head
        self.reduced_channels_reward = 128  # Number of channels in reward head
        self.reduced_channels_value = 128  # Number of channels in value head
        self.reduced_channels_policy = 128  # Number of channels in policy head
        # Define hidden layers (example)
        self.resnet_fc_reward_layers = [256, 256]  # Hidden layers for reward head
        self.resnet_fc_value_layers = [256, 256]  # Hidden layers for value head
        self.resnet_fc_policy_layers = [256, 256]
        # Hidden layers for policy head # Define the hidden layers in the policy head of the prediction network
        self.support_size = 15  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size
        self.downsample = "resnet"  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)

        ### Training  训练相关参数
        # 训练日志和相关数据保存地址
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(
            __file__).stem / datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        # 保存模型权重
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        # 整体训练轮次
        self.training_steps = 20000  # Total number of training steps (ie weights update according to a batch)
        # batch size大小
        self.batch_size = 512  # Number of parts of games to train on at each training step
        # 多少轮保存一次数据
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        """
        总结：
        初期稳定性不佳: 如果模型在训练的早期表现出不稳定的情况，可以稍微增大 value_loss_weight 来减轻这种波动。
        后期细调: 在训练的中后期，逐步调高 value_loss_weight，以确保价值预测的稳定性，并减少训练过程中的波动。
        """
        self.value_loss_weight = 1.0  # 缩放value loss避免过拟合,论文参数是0.25
        self.entropy_loss_weight = 0.05  # 缩放entropy_loss
        """
        # 初期阶段: 增大 entropy_loss_weight 以增强探索性，帮助模型更好地适应复杂环境。
        # 中后期阶段: 减小 entropy_loss_weight 以加快收敛，减少训练过程中的波动。
        # 平衡策略: 根据具体任务需求，找到适中的配置，确保训练过程的稳定性和策略的有效性。
        """
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.optimizer = "AdamW"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0001  # Initial learning rate
        self.lr_decay_rate = 0.95  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 5000  # 缓存空间中记录的自我监督的数据数量,给高了的话,容易引入噪声,如果给低了,性能不佳不稳定
        """
        举个例子：
        假设你在训练一个自动驾驶模型，在模拟中车辆经过一个弯道。设置 self.num_unroll_steps = 15 
        意味着模型会同时考虑到进入弯道的前 15 步以及驶出弯道的后 15 步的所有状态和决策。
        这有助于模型理解如何在弯道中保持最佳车速和方向。
        总结：
        self.num_unroll_steps 决定了模型在每个训练批次中要展开多少步（或状态转换）。
        这个参数的配置对模型捕捉时间相关性和优化长期决策非常关键。
        选择合适的 num_unroll_steps 可以帮助模型更好地理解和预测未来的状态和奖励，从而提升训练效果和决策质量。
        """
        self.num_unroll_steps = 15  # 每个批次中保留多少数量的moves的数据
        """
        例子：
        假设在一个自动驾驶任务中，车辆需要计划如何通过一个复杂的交通路口。设置 td_steps 为 5 意味着模型将根据未来的 
        5 个状态及其奖励来估计当前决策的价值。这可以帮助模型更好地权衡在复杂路口中短期和长期的风险与收益。
        总结：
        td_steps 是一个控制时间差分更新步数的参数，用于决定在计算当前状态的目标价值时，要向未来看多少步。
        它影响了模型在短期与长期回报之间的权衡，配置合适的 td_steps 对于提升模型的表现至关重要。
        """
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        """
        这两个参数 `self.PER` 和 `self.PER_alpha` 与**优先经验回放（Prioritized Experience Replay, PER）**相关，这是强化学习中用于提高样本效率和加快收敛的一种技术。
        ### 1. **`self.PER`**: 
           - **定义**: `self.PER` 是一个布尔值，用于控制是否启用优先经验回放机制。
           - **作用**:
             - **启用优先经验回放**: 当 `self.PER = True` 时，模型会在回放经验（从经验回放缓冲区中采样经验用于训练）时，优先选择那些对模型来说更“意外”的样本进行学习。这些意外样本通常是模型预测误差较大的样本，可能包含对模型更有价值的学习信息。
             - **标准经验回放**: 如果 `self.PER = False`，模型则会使用标准的经验回放，随机均匀地从经验回放缓冲区中采样样本进行训练。这样每个样本被选中的概率相同，不考虑样本的重要性。

           - **优先经验回放的好处**:
             - 加速学习：通过优先学习那些对当前策略影响最大的经验，可以加速模型的学习进程。
             - 更高效的样本利用：避免模型浪费时间在那些对当前策略提升作用较小的样本上。

        ### 2. **`self.PER_alpha`**:
           - **定义**: `self.PER_alpha` 控制优先经验回放中的**优先化程度**，其值在 0 到 1 之间。
           - **作用**:
             - **优先化程度**: `PER_alpha` 决定了优先经验回放中样本被选中的优先化程度。值为 0 时，优先经验回放退化为均匀随机采样，即没有优先化；值为 1 时，优先化程度达到最大，样本被选中的概率完全由它们的优先级（通常是 TD 误差）决定。
             - **影响采样策略**: 当 `self.PER_alpha` 较高时（接近1），模型将更倾向于选择那些 TD 误差大的样本进行学习，这些样本通常对当前策略的改进更有价值；当 `self.PER_alpha` 较低时（接近0），模型将更接近于随机采样，从而保持采样的多样性。

           - **配置建议**:
             - **高优先化（`PER_alpha` 接近1）**: 更适合在稳定且成熟的模型阶段，以最大化利用那些难以预测或误差大的样本。
             - **低优先化（`PER_alpha` 接近0）**: 更适合在训练初期，以确保模型对环境有更全面的探索，避免陷入局部最优。

        ### 例子：
        假设你在训练一个自动驾驶模型，其中一些状态-动作对（如紧急刹车）对策略的改进至关重要。如果 `self.PER = True` 且 `self.PER_alpha` 接近 1，模型会更频繁地采样这些关键状态，以加快学习速度并提高模型对这些关键决策的准确性。

        ### 总结：
        - **`self.PER`**：控制是否启用优先经验回放，启用后模型会更频繁地学习那些对其策略影响最大的样本。
        - **`self.PER_alpha`**：控制优先经验回放中的优先化程度，值越高，样本的选择越依赖其优先级，有助于更高效地利用经验样本。
        """
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.6  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """控制完整训练过程中的温度阈值
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1
        elif trained_steps < 0.75 * self.training_steps:
            return 0.25
        else:
            return 0.01


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make('highway-v0', render_mode="rgb_array",
                            config={  # 需要在程序启动这个观测器之前使用自定义的公式来对观测车辆的初始速度和初始位置进行初始化
                                'observation': {"type": "Kinematics",  # 使用这个观测器作为状态空间，可以获取观测车辆位置、观测车辆速度和观测车辆转向角
                                                "vehicles_count": 21,  # 20辆周围车辆
                                                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                                                # 控制状态空间包括转向角
                                                "absolute": True,  # 使用相对坐标，相对于观测车辆。为True时使用相对于环境的全局坐标系。
                                                "order": "sorted"  # 根据与自车的距离从近到远排列。这种排列方式使得观测数组的顺序保持稳定
                                                },
                                'action': {'type': 'ContinuousAction',
                                           'acceleration_range': (-4, 4.0)},  # 为它扩展一个能够控制横向加速度和纵向加速度的子类
                                'simulation_frequency': 5,  # 模拟频率
                                'policy_frequency': 5,  # 策略频率
                                # 纵向决策：IDM（智能驾驶模型）根据前车的距离和速度计算出加速度。
                                'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                                'screen_width': 600,  # 屏幕宽度
                                'screen_height': 150,  # 屏幕高度
                                'centering_position': [0.3, 0.5],  # 初始缩放比例
                                'scaling': 5.5,  # 偏移量
                                'show_trajectories': False,  # 是否记录车辆最近的轨迹并显示
                                'render_agent': True,  # 控制渲染是否应用到屏幕
                                'offscreen_rendering': False,  # 当前的渲染是否是在屏幕外进行的。如果为False，意味着渲染是在屏幕上进行的，
                                'manual_control': False,  # 是否允许键盘控制观测车辆
                                'real_time_rendering': False,  # 是否实时渲染画面
                                'lanes_count': 4,  # 车道数量
                                # 'normalize_reward': True,
                                'controlled_vehicles': 1,  # 一次只控制一辆车
                                'initial_lane_id': None,  # 控制观测车辆在哪一条车道初始化
                                'duration': 30,  # 限制了仿真的时间长度
                                'ego_spacing': 1.5,  # 表示控制车辆（ego vehicle）与前一辆车之间的初始间隔距离。它用来设置在创建控制车辆时的车间距
                                'vehicles_density': 1,
                                "right_lane_reward": 0.5,  # 在最右边的车道上行驶时获得的奖励，在其他车道上线性映射为零。
                                'collision_reward': -1.5,  # 与车辆相撞时获取的惩罚
                                'on_road_reward': 1.5,  # 在路上正常行驶的奖励
                                'high_speed_reward': 1.5,  # 维持高速行驶的奖励
                                'lane_change_reward': -0.5,  # 换道的惩罚
                                'reward_speed_range': [20, 30],  # 高速的奖励从这个范围线性映射到[0,HighwayEnv.HIGH_SPEED_REWARD]。
                                'offroad_terminal': True  # 车辆偏离道路是否会导致仿真结束
                            })
        self.seed = seed
        self.env.reset()
        self.gif_imgs = []

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        # logger.info(f"start step: {datetime.datetime.now()}")
        action = numpy.tanh(action)
        observation, reward, done, _, _ = self.env.step(action)
        # observation = observation.reshape((147,))

        # logger.info(f"end step: {datetime.datetime.now()}")
        return numpy.array([observation]), reward, done

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation, info = self.env.reset(seed=self.seed)

        # Reshape the observation to 3D (1, 1, -1)
        observation = np.array(observation)
        observation = observation.reshape((1, 21, 7))
        # logger.info(f"Observation2 reset shape after step:{type(observation)}-shape-{observation.shape}")
        return observation

    def render(self):
        """
        Display the game observation.
        """
        logger.info(f"start render step: {datetime.datetime.now()}")
        self.env.render()
        logger.info(f"end render step: {datetime.datetime.now()}")
        time.sleep(3)
