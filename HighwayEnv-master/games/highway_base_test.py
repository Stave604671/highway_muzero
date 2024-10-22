import datetime
import pathlib
import gymnasium as gym
import numpy
import torch
import numpy as np
from ray import logger
import highway_env   # 不加这个的话会进不了环境
from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        self.seed = 0  # 随机数种子,用于固定随机性方便复现
        self.max_num_gpus = None  # 固定使用gpu的最大数量.使用单个gpu会更快,没有配置的话会默认使用所有gpu

        # Game
        self.observation_shape = (1, 21, 3)  # 游戏观测空间的维度,如果观测空间三维无所谓,如果是一维,需要配成(1,1,x)
        self.action_space = 2  # 动作空间的大小
        self.players = [i for i in range(1)]  # 玩家的数量,车辆换道场景观测和控制车辆只有一个,为1就行
        self.stacked_observations = 0  # 观测时叠加的历史观察数量（包括过去的动作）。

        # Evaluate
        self.muzero_player = 0  # 用于区分多玩家环境中谁是 MuZero 控制的玩家。自我对弈默认是0.
        self.opponent = None  # MuZero 面对的对手，用于评估在多人游戏中的进展。可以是 "random" 或 "expert"，如果在游戏类中实现了对手

        # Self-Play
        self.num_workers = 1  # 定义了同时进行 Self-Play 的工作线程数量，这些线程负责生成训练样本并将其存储到回放缓冲区中。
        self.selfplay_on_gpu = True  # 是否在gpu进行自我博弈,打开后速度变快,但是显存开支会高很多
        self.max_moves = 500  # 每场游戏的最大游戏次数,未发生碰撞,或者没有达到这个次数,单场游戏都不停止
        self.num_simulations = 30  # 执行指定次数的模拟，每次模拟从根节点开始进行搜索和更新,
        self.discount = 0.997  # 长期回报的折扣因子
        self.temperature_threshold = None  # 单次play_games的温度阈值,当前的play_games内,最大移动self.max_moves次,moves的次数超过这个阈值后,温度直接为0,低于这个次数时,启用visit_softmax_temperature_fn获取温度数值
        # 'uniform' or 'density'
        self.node_prior = 'uniform'

        # UCB formula
        self.pb_c_base = 19652  # 数值越大,更倾向于利用选择已知效果较好的动作,而非探索新动作
        self.pb_c_init = 1.25  # 初始化参数,对探索奖励有一个固定的提升作用.数值越大,初期的探索越多.反之更依赖已知动作

        # Progressive widening parameter
        self.pw_alpha = 0.49

        # network_config2
        self.network = "fullyconnected"
        self.log_std_clamp = (-20, 2)  # Clamp the standard deviation
        self.encoding_size = 10
        # Fully Connected Network
        self.fc_reward_layers = [64, 64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64, 64]  # Define the hidden layers in the value network
        self.fc_mu_policy_layers = [64, 64]  # Define the hidden layers in the policy network
        self.fc_log_std_policy_layers = [64, 64]  # Define the hidden layers in the policy network
        self.fc_representation_layers = [64, 64]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64, 64]  # Define the hidden layers in the dynamics network
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

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
        self.batch_size = 128  # Number of parts of games to train on at each training step
        # 多少轮保存一次数据
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # 缩放value loss避免过拟合,论文参数是0.25,直接给到五倍好了
        self.entropy_loss_weight = 0.10  # 缩放entropy_loss
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.optimizer = "AdamW"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.004  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 500  # 缓存空间中记录的自我监督的数据数量,给高了的话,容易引入噪声,如果给低了,性能不佳不稳定

        self.num_unroll_steps = 20  # 每个批次中保留多少数量的moves的数据

        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value

        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True   # windows下需要都打开，linux没限制

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 1/100  # Number of seconds to wait after each played game
        self.training_delay = 1/100  # Number of seconds to wait after each training step
        self.ratio = 1/100  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
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
        self.env = gym.make('highway-fast-v0', render_mode="rgb_array",
                            config={  # 需要在程序启动这个观测器之前使用自定义的公式来对观测车辆的初始速度和初始位置进行初始化
                                'observation': {"type": "Kinematics",  # 使用这个观测器作为状态空间，可以获取观测车辆位置、观测车辆速度和观测车辆转向角
                                                "vehicles_count": 21,  # 20辆周围车辆
                                                "features": ["presence", "x", "y"],
                                                # "features_range": {
                                                #     # "x": [-100, 100],
                                                #     # "y": [-100, 100],
                                                #     "vx": [-30, 30],
                                                #     "vy": [-30, 30]
                                                # },
                                                # 控制状态空间包括转向角
                                                "absolute": True,  # 使用相对坐标，相对于观测车辆。为True时使用相对于环境的全局坐标系。
                                                "order": "sorted",
                                                "normalize": False,# 根据与自车的距离从近到远排列。这种排列方式使得观测数组的顺序保持稳定
                                                },
                                # 'action': {'type': 'DiscreteMetaAction'},
                                'action': {'type': 'ContinuousAction',
                                           'acceleration_range': (-4, 4.0),
                                           'steering_range': (-np.pi / 12, np.pi / 12)},  # 为它扩展一个能够控制横向加速度和纵向加速度的子类
                                'simulation_frequency': 24,  # 模拟频率
                                'policy_frequency': 24,  # 策略频率
                                # 纵向决策：IDM（智能驾驶模型）根据前车的距离和速度计算出加速度。
                                'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                                'screen_width': 900,  # 屏幕宽度
                                'screen_height': 600,  # 屏幕高度
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
                                "right_lane_reward": 0.2,  # 在最右边的车道上行驶时获得的奖励，在其他车道上线性映射为零。
                                'collision_reward': -1.5,  # 与车辆相撞时获取的惩罚
                                'high_speed_reward': 3,    # 维持高速行驶的奖励
                                'lane_change_reward': -0.2,  # 换道的惩罚
                                'reward_speed_range': [20, 30],  # 高速的奖励从这个范围线性映射到[0,HighwayEnv.HIGH_SPEED_REWARD]。
                                'offroad_terminal': True  # 车辆偏离道路是否会导致仿真结束
                            })
        self.seed = seed
        self.env.reset()
        self.previous_positions = None  # 存储上一个时间步的车辆位置
        self.dt = 1 / 12  # 时间步长 (1 / policy_frequency)
        # self.velocities = 25
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
        # action = numpy.tanh(action)
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
        observation = observation.reshape((1, 21, 3))
        # logger.info(f"Observation2 reset shape after step:{type(observation)}-shape-{observation.shape}")
        return observation

    def render(self):
        """
        Display the game observation.
        """
        # logger.info(f"start render step: {datetime.datetime.now()}")
        self.env.render()
        # logger.info(f"end render step: {datetime.datetime.now()}")
        # time.sleep(3)

    def render_rgb(self):
        rgb_img = self.env.render()
        self.gif_imgs.append(rgb_img)

    def save_gif(self):
        # imageio.mimsave(
        #     f'./{datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}.gif',
        #     self.gif_imgs,
        #     fps=5,
        # )
        self.gif_imgs = []

    def close(self):
        self.env.close()