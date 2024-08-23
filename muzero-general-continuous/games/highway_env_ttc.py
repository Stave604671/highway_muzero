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
        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available


        ### Game
        self.observation_shape = (1, 1, 147)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = 2  # Number of dimensions in the action space
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 3  # Number of simultaneous threads self-playing to feed the replay buffer
        self.selfplay_on_gpu = False  # 启用渲染需要把它打开
        self.max_moves = 1000  # Maximum number of moves if game is not finished before
        self.num_simulations = 35  # Number of future moves self-simulated
        self.discount = 0.975  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping temperature to 0 (ie playing according to the max)
        self.node_prior = 'uniform'  # 'uniform' or 'density'

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.5
        self.root_exploration_fraction = 0.5

        # UCB formula
        self.pb_c_base = 19000
        self.pb_c_init = 1.25

        # Progressive widening parameter
        self.pw_alpha = 0.4

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size
        
        # Residual Network
        self.downsample = "resnet" # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 16  # Number of blocks in the ResNet
        self.channels = 256  # Number of channels in the ResNet
        # Define channels for each head
        self.reduced_channels_reward = 256  # Number of channels in reward head
        self.reduced_channels_value = 256  # Number of channels in value head
        self.reduced_channels_policy = 256  # Number of channels in policy head

        # Define hidden layers (example)
        self.resnet_fc_reward_layers = [256, 256]  # Hidden layers for reward head
        self.resnet_fc_value_layers = [256, 256]  # Hidden layers for value head
        self.resnet_fc_policy_layers = [256, 256]
        # Hidden layers for policy head # Define the hidden layers in the policy head of the prediction network
        # self.resnet_fc_reconstruction_layers = [32]  # Define the hidden layers in the reconstruction head of the reconstruction network

        # Fully Connected Network
        self.encoding_size = 40
        self.fc_representation_layers = [256, 256]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [256, 256]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [256, 256]  # Define the hidden layers in the reward network
        self.fc_value_layers = [512, 512]  # Define the hidden layers in the value network
        self.fc_mu_policy_layers = [128, 128]  # Define the hidden layers in the policy network
        self.fc_log_std_policy_layers = [128, 128]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 20000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256 # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.entropy_loss_weight = 0.1  # Scale the entropy loss
        self.log_std_clamp = (-20, 2)  # Clamp the standard deviation
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.0003  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 5000



        ### Replay Buffer
        self.replay_buffer_size = 5000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 15  # Number of game moves to keep for every batch element
        self.td_steps = 35  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.6 # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1
        elif trained_steps < 0.75 * self.training_steps:
            return 0.1
        else:
            return 0.01


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make('highway-v0', render_mode="rgb_array", config={    # 需要在程序启动这个观测器之前使用自定义的公式来对观测车辆的初始速度和初始位置进行初始化
                'observation': {"type": "Kinematics",  # 使用这个观测器作为状态空间，可以获取观测车辆位置、观测车辆速度和观测车辆转向角
                                "vehicles_count": 21,  # 20辆周围车辆
                                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],  # 控制状态空间包括转向角
                                "absolute": True,  # 使用相对坐标，相对于观测车辆。为True时使用相对于环境的全局坐标系。
                                "order": "sorted"   # 根据与自车的距离从近到远排列。这种排列方式使得观测数组的顺序保持稳定
                                },
                'action': {'type': 'ContinuousAction',
                           'acceleration_range': (-4, 4.0)},  # 为它扩展一个能够控制横向加速度和纵向加速度的子类
                'simulation_frequency': 35,  # 模拟频率
                'policy_frequency': 5,  # 策略频率
                # 纵向决策：IDM（智能驾驶模型）根据前车的距离和速度计算出加速度。
                'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                'screen_width': 600,  # 屏幕宽度
                'screen_height': 150,  # 屏幕高度
                'centering_position': [0.3, 0.5],  # 初始缩放比例
                'scaling': 5.5,  # 偏移量
                'show_trajectories': False,  # 是否记录车辆最近的轨迹并显示
                'render_agent': False,  # 控制渲染是否应用到屏幕
                'offscreen_rendering': False,  # 当前的渲染是否是在屏幕外进行的。如果为False，意味着渲染是在屏幕上进行的，
                'manual_control': False,  # 是否允许键盘控制观测车辆
                'real_time_rendering': False,  # 是否实时渲染画面
                'lanes_count': 4,  # 车道数量
                # 'normalize_reward': True,
                'controlled_vehicles': 1,  # 一次只控制一辆车
                'initial_lane_id': None,  # 控制观测车辆在哪一条车道初始化
                'duration': 60,  # 限制了仿真的时间长度
                'ego_spacing': 2,  # 表示控制车辆（ego vehicle）与前一辆车之间的初始间隔距离。它用来设置在创建控制车辆时的车间距
                'vehicles_density': 1,
                "right_lane_reward": 0.1,  # 在最右边的车道上行驶时获得的奖励，在其他车道上线性映射为零。
                'collision_reward': -5,  # 与车辆相撞时获取的奖励
                'on_road_reward': 5,
                # 'high_speed_reward': 0.4,
                'lane_change_reward': -1,
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
        # action = numpy.tanh(action)
        observation, reward, done, _, _ = self.env.step(action)
        observation = observation.reshape((147,))

        # logger.info(f"end step: {datetime.datetime.now()}")
        return numpy.array([[observation]]), reward, done

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        observation, info = self.env.reset(seed=self.seed)

        # Reshape the observation to 3D (1, 1, -1)
        observation = np.array(observation)
        observation = observation.reshape((1, 1, 147))
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
