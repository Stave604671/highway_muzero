import datetime
import math
import time
import numpy as np
import numpy
import ray
import torch
from ray import logger
import models


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        # 读入game路径下的配置信息
        self.config = config
        # 初始化游戏环境
        self.game = Game(seed)
        # 固定生成的随机数种子，确保结果可以稳定复现
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        # 初始化自我对弈的网络
        # logger.info(f"{torch.cuda.is_available()}--cudause1")
        self.model = models.MuZeroNetwork(self.config)
        # logger.info(f"device1 {next(self.model.parameters()).device}")
        # 权重初始化
        self.model.set_weights(initial_checkpoint["weights"])
        # logger.info(f"device2 {next(self.model.parameters()).device}")
        # 控制当前网络是否使用gpu，如果启用，一些设备可能负担不了
        # self.model.to(torch.device("cpu"))
        # logger.info(f"{torch.cuda.is_available()}--cudause2")
        # self.model.to("cuda:0")
        self.model.to(torch.device("cuda:0" if self.config.selfplay_on_gpu else "cpu"))
        # 对模型启用验证模式
        logger.info(f"device3 {next(self.model.parameters()).device}")
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        """
        :param shared_storage: 分布式环境中用于共享信息的对象
        :param replay_buffer: 存储对弈历史的缓冲区，用于后续训练
        :param test_mode: 指示当前是否为测试模式。如果为True，则以确定性方式进行游戏（没有探索）；否则，按照常规策略进行探索。
        """
        # 当训练步数未结束、并且训练状态也未被终止，持续循环对权重进行更新
        while ray.get(
                shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")  # 训练被终止的标记
        ):
            # 更新当前的模型权重
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))
            # 训练模式
            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )
                # 游戏结束后，将生成的游戏历史保存到 replay_buffer 中，用于后续训练。
                replay_buffer.save_game.remote(game_history, shared_storage)
            # 测试模式
            else:
                # Take the best action (no exploration) in test mode
                logger.info(f"test begin")
                game_history = self.play_game(
                    0,  # 测试模式下不进行探索，直接选择最优动作
                    self.config.temperature_threshold,  # 一定步数后控制未0，此时好像也不是很重要
                    False,  # 是否渲染画面
                    # 只有一个玩家默认自我对弈，否则指定玩家
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )
                # 保存测试结果，包括回合长度、总奖励、平均根节点值等。
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                        "muzero_reward": sum(game_history.reward_history),
                        "opponent_reward": 0,  # 如果没有对手，opponent_reward 设置为 0
                    }
                )
                # print("self.config.players:",self.config.players)
                # if 1 < len(self.config.players):
                #     shared_storage.set_info.remote(
                #         {
                #             "muzero_reward": sum(
                #                 reward
                #                 for i, reward in enumerate(game_history.reward_history)
                #                 if game_history.to_play_history[i - 1]
                #                 == self.config.muzero_player
                #             ),
                #             "opponent_reward": sum(
                #                 reward
                #                 for i, reward in enumerate(game_history.reward_history)
                #                 if game_history.to_play_history[i - 1]
                #                 != self.config.muzero_player
                #             ),
                #         }
                #     )

            # Managing the self-play / training ratio
            # 在两次self - play之间延迟，以控制自我对弈的频率。
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            """
            目的：这段代码的目的是通过控制自我对弈的进度，来平衡自我对弈和模型训练的步数。具体来说，
            它通过检查当前的训练步数与自我对弈步数之间的比例，并在比例过小（即训练步数增长太快、自我对弈步数跟不上）时，
            暂时停止训练，让自我对弈有时间追赶。
            工作原理：
                在满足条件的情况下，程序会持续检查训练步数和自我对弈步数之间的比例。如果比例小于设定的 self.config.ratio，
            则程序会暂停 0.5 秒，然后继续检查，直到这个比例满足要求或者达到终止条件。这样做可以避免模型在自我对弈数据不足的情况
            下进行过度训练，确保训练数据的质量和模型训练的效果。
            """
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)
        # 关闭游戏清理游戏状态
        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """ 使用蒙特卡洛树搜索（MCTS）来模拟一场游戏对局
        Play one game with actions based on the Monte Carlo tree search at each moves.
        :param temperature: 训练温度
        :param temperature_threshold: 温度阈值，当训练步数超过temperature_threshold时，训练温度直接归0
        :param render: 是否渲染游戏画面。如果为 True，会在每一步显示游戏状态，用于调试或观察对局情况。训练过程默认是false
        :param opponent: 对手的类型，默认是self，也就是自我对弈
        :param muzero_player: 当前的 MuZero 玩家，用于区分多玩家环境中谁是 MuZero 控制的玩家。自我对弈默认是0.
        """
        # 初始化游戏历史记录和初始观测
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(numpy.zeros(self.config.action_space))
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())
        done = False  # 检查游戏是否结束
        mcts = MCTS(self.config)
        # logger.info("value1")
        if render:  # 是否开启画面渲染
            # logger.info("value2")
            self.game.render()
            # logger.info("value3")
        # 禁用 PyTorch 的梯度计算，确保代码在推理时不计算梯度，以提高速度和减少内存消耗。
        def get_lane(y_coord):
            if 0 <= y_coord < 4:
                return 1
            elif 4 <= y_coord < 8:
                return 2
            elif 8 <= y_coord < 12:
                return 3
            elif 12 <= y_coord < 16:
                return 4
            return -1  # 超出车道范围
        with torch.no_grad():
            while (  # 开始一个循环，直到游戏结束 (done=True) 或者达到最大移动次数
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                # 观测空间的返回值需要是三维的
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                # 观测空间的大小需要与配置文件一致
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                # 用于获取堆叠的观测数据。通常用于提供过去几帧的观测数据，以帮助模型做出更好的决策。【实验验证连续型场景这个参数对训练没帮助】
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations
                )
                # Choose the action如果 opponent 是 "self" 或者当前轮到的玩家是 muzero_player，则通过蒙特卡洛树搜索（MCTS）选择行动
                if opponent == "self" or muzero_player == self.game.to_play():
                    # 初始化蒙特卡洛树，并返回根节点和蒙特卡洛树的其他信息
                    logger.info(f"初始化树{datetime.datetime.now()}")
                    root, mcts_info = mcts.run(
                        self.model,
                        stacked_observations,
                        self.game.to_play(),
                        render,
                    )
                    # 根据 MCTS 的结果遍历树选择一个action
                    # logger.info(f"单步规划时间1：{datetime.datetime.now()}")
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                           or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )
                    obs_car_line_value = observation[0][0][2]
                    # 获取观测车辆的 y 坐标（第一行的第三列）
                    ego_vehicle_y = observation[0, 0, 2]
                    ego_lane = get_lane(ego_vehicle_y)
                    # 过滤出存在于观测空间的车辆（第一列为 1）
                    present_cars = observation[0, observation[0, :, 0] == 1]
                    # 检查是否存在与观测车辆在同一车道的车辆
                    same_lane_exists = any(get_lane(car[2]) == ego_lane for car in present_cars)
                    if obs_car_line_value < 4 and action.value[1] < 0:  # 在第一车道并且尝试左拐，改成右拐
                        action.value[1] = -action.value[1]
                    if obs_car_line_value > 12 and action.value[1] > 0:  # 在第四车道并且尝试右拐，改成左拐
                        action.value[1] = -action.value[1]
                    if not same_lane_exists:  # 没车辆和它在相同车道，不拐
                        action.value[1] = 0
                    logger.info(f"{observation}----{action.value}")
                    # logger.info(f"单步规划时间2：{datetime.datetime.now()}")
                    # logger.info(f"观测车辆车道坐标{obs_car_line_value} 观测车辆的车道角度2：{type(action.value[1])}{action.value[1]}")
                    if render:  # 渲染模式打印日志结果
                        logger.info(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        logger.info(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
                else:
                    # 如果当前玩家不是 muzero_player，则调用选择对手的动作
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )
                # 执行蒙特卡洛树选择的动作，返回新的观测 observation、当前获得的奖励 reward，以及游戏是否结束的标志 done。
                observation, reward, done = self.game.step(action.value)
                if render:  # 渲染模式下实时打印当前状态
                    logger.info(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()
                # 存储当前 MCTS 搜索的统计信息，比如各个动作的访问次数等。
                game_history.store_search_statistics(root)
                # Next batch 将当前的动作、观测、奖励和下一轮的玩家信息分别添加到游戏历史记录中。
                game_history.action_history.append(action.value)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())
        self.game.render_rgb()
        self.game.save_gif()
        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            """
            opponent == "human": 如果对手是人类，执行以下操作：运行蒙特卡洛树搜索（MCTS），使用当前的观测数据 stacked_observations 
            和模型来评估最佳动作。打印树的最大深度以及根节点的价值。打印 MuZero 推荐的动作。self.game.human_to_action() 
            用于获取人类玩家选择的动作。返回玩家选择的动作和 MCTS 的根节点 root。
            """
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.to_play(),
                True,
            )
            action_str = self.game.action_to_string(self.select_action(root, 0))
            logger.info(f'Tree depth: {mcts_info["max_tree_depth"]}')
            logger.info(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            logger.info(f"Player {self.game.to_play()} turn. MuZero suggests {action_str}")
            return self.game.human_to_action(), root
        elif opponent == "expert":
            # 如果对手是专家，调用 self.game.expert_agent() 获取专家对手的动作。返回专家的动作和 None（因为专家模式不需要 MCTS 信息）。
            return self.game.expert_agent(), None
        elif opponent == "random":
            # 如果对手是随机的，执行以下操作：
            # 检查合法动作列表 self.game.legal_actions() 不为空。
            # 确保合法动作是动作空间的子集。
            # 从合法动作中随机选择一个动作。
            # 返回随机选择的动作和 None（因为随机对手不需要 MCTS 信息）。
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        :param node: MCTS 树中的一个节点。节点包含子节点，每个子节点代表一个可能的动作。
        :param temperature: 控制探索和利用的参数。温度越高，探索的概率越大；温度越低，选择访问次数最多的动作的概率越高。
        """
        # 从每个子节点中提取访问计数，构成一个numpy数组 这些访问计数表示每个动作被选择的频率。
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        # 存储所有可能的动作，这些动作是节点的子节点对应的键。
        actions = [action for action in node.children.keys()]
        # 当温度为零时，选择访问次数最多的动作。这里使用 numpy.argmax(visit_counts) 找到访问次数最多的动作的索引。
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        # 当温度为无穷大时，选择一个随机动作。这种情况下，所有动作的选择概率是均等的，使用 numpy.random.choice(actions) 随机选择一个动作。
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            # 将访问计数通过温度进行缩放。这是为了计算加权分布，其中温度较低时，较高的访问计数将获得更多的权重。
            visit_count_distribution = visit_counts ** (1 / temperature)
            # 将权重归一化为概率分布，使它们的总和为 1。
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            # 根据计算出的概率分布选择一个动作。
            action = numpy.random.choice(actions, p=visit_count_distribution)
        return action


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(self, model, observation, to_play, render, override_root_with=None):
        """ 执行mcts搜索
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        :param model: 用于估算值和奖励的模型。
        :param observation: 当前环境的观察结果。
        :param to_play: 当前轮到哪个玩家。
        :param render: 是否渲染游戏画面。
        :param override_root_with: 如果提供，则使用此节点作为根节点。
        """
        if override_root_with:  # 如果 override_root_with 存在，则使用它作为根节点，否则从观察中创建一个新节点并用模型进行初始推断。
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node()
            # 将输入的 observation 转换为 PyTorch 张量，并将其传输到与模型相同的设备（CPU 或 GPU）。
            observation = (torch.tensor(observation).float().unsqueeze(0).to(next(model.parameters()).device))
            # 使用模型对当前观察进行初始推断，得到根节点的预测值、奖励、策略均值、策略对数标准差和隐藏状态。
            (root_predicted_value, reward, mu, log_std, hidden_state, ) = model.initial_inference(observation)
            # if render:
            #     logger.info("input", observation)
            #     logger.info("root_predicted_value", models.support_to_scalar(root_predicted_value,
            #                                                                  self.config.support_size),)
            #     logger.info("mu", mu)
            #     logger.info("sigma", log_std)
            # 将预测值从支持空间转换为标量值，并将其转为 Python 标量（浮点数）。
            root_predicted_value = models.support_to_scalar(root_predicted_value, self.config.support_size).item()
            # 将奖励从支持空间转换为标量值，并将其转为 Python 标量（浮点数）。
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            # 扩展根节点，设置其 to_play、reward、mu、log_std 和 hidden_state 属性。
            root.expand(to_play, reward, mu, log_std, hidden_state,)
        # 初始化一个 MinMaxStats 实例，用于在搜索过程中跟踪奖励和价值的最小值和最大值，以便在更新节点时进行归一化处理。
        min_max_stats = MinMaxStats()
        # 初始化 max_tree_depth 变量，用于记录搜索过程中遇到的最大树深度。
        max_tree_depth = 0
        # 执行指定次数的模拟（self.config.num_simulations），每次模拟从根节点开始进行搜索和更新。
        for _ in range(self.config.num_simulations):
            # 初始化 virtual_to_play 变量，表示当前轮到哪个玩家。在每次模拟中，玩家会轮流进行操作。
            virtual_to_play = to_play
            # 将 node 设置为当前的根节点，以开始搜索。
            node = root
            # 初始化 search_path 列表，用于跟踪当前模拟中的搜索路径。
            search_path = [node]
            # 初始化 current_tree_depth 变量，表示当前模拟中的树深度。
            current_tree_depth = 0
            # 遍历当前节点的子节点，直到找到一个未扩展的节点。
            while node.expanded():
                # 更新当前树深度
                current_tree_depth += 1
                # 从当前节点选择一个子节点和对应的动作，并将该子节点设置为当前节点。
                # select_child 方法会根据 UCB (Upper Confidence Bound) 算法来选择子节点。
                action, node = self.select_child(node, min_max_stats)
                # 将选择的子节点添加到搜索路径中。
                search_path.append(node)
                # 更新 virtual_to_play，表示轮到下一个玩家。如果当前玩家是最后一个玩家，则回到第一个玩家。
                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            # 获取当前节点的父节点，以便进行后续的状态更新。
            parent = search_path[-2]
            # 使用模型的 recurrent_inference 方法进行推断，以获取下一个状态的值、奖励、策略均值、策略对数标准差和隐藏状态。
            value, reward, mu, log_std, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor(numpy.array([action.value]), dtype=torch.float32).to(
                    parent.hidden_state.device
                ),
            )
            # 将预测的值从支持空间转换为标量值，并将其转为 Python 标量（浮点数）。
            value = models.support_to_scalar(value, self.config.support_size).item()
            # 将预测的奖励从支持空间转换为标量值，并将其转为 Python 标量（浮点数）。
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            # 扩展当前节点，使用从模型推断得到的 virtual_to_play、reward、mu、log_std 和 hidden_state 更新节点的状态。
            # expand 方法会将这些信息用于生成子节点。
            node.expand(
                virtual_to_play,
                reward,
                mu,
                log_std,
                hidden_state,
            )
            # 通过调用 backpropagate 方法，将模拟结果沿搜索路径回传并更新节点的统计数据。
            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)
            # 更新 max_tree_depth，记录当前模拟过程中遇到的最大树深度。
            max_tree_depth = max(max_tree_depth, current_tree_depth)
        # 创建一个字典 extra_info，用于存储额外的信息，例如最大树深度和根节点的预测值。这些信息可以用于调试和分析。
        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """ 选择具有最高 UCB 评分的子节点。
        Select the child with the highest UCB score.
        """
        # Progressive widening (See https://hal.archives-ouvertes.fr/hal-00542673v2/document)
        # 判断当前节点的子节点数量是否小于 如果是，采用渐进扩展策略。
        if len(node.children) < (node.visit_count + 1) ** self.config.pw_alpha:
            # 从以 node.mu 为均值、node.std 为标准差的正态分布中采样一个动作值，并将其转换为 NumPy 数组。
            distribution = torch.distributions.normal.Normal(node.mu, node.std)
            action_value = distribution.sample().squeeze(0).detach().cpu().numpy()
            # 确保采样的动作值不会重复。即，采样的动作必须不在当前节点的子节点中。
            while Action(action_value) in node.children.keys():
                action_value = distribution.sample().squeeze(0).detach().cpu().numpy()
            # 将采样的动作值转换为 Action 对象，并在当前节点的 children 字典中创建一个新的子节点。返回该动作及对应的子节点。
            action = Action(action_value)
            node.children[action] = Node()
            return action, node.children[action]

        else:
            # 计算所有子节点的 UCB 评分，并找到最大的 UCB 评分。
            max_ucb = max(
                self.ucb_score(node, child, min_max_stats)
                for action, child in node.children.items()
            )
            # 从具有最高 UCB 评分的子节点中随机选择一个动作。numpy.random.choice 从满足条件的动作中进行选择。
            action = numpy.random.choice(
                [
                    action
                    for action, child in node.children.items()
                    if self.ucb_score(node, child, min_max_stats) == max_ucb
                ]
            )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """ 计算子节点的 UCB（Upper Confidence Bound）评分，以决定哪个子节点应该被扩展或访问。UCB 评分是节点值和探索奖励的组合。
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        # pb_c 是一个探索奖励的系数，基于父节点的访问次数、一个基础常数 pb_c_base 和一个初始化常数 pb_c_init 计算。
        # math.log 和 math.sqrt 用于调整探索奖励的强度，以平衡探索和利用。
        # parent.visit_count 是父节点的访问次数，child.visit_count 是子节点的访问次数。
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        # 计算探索奖励
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        # 计算先验奖励
        # Uniform prior for continuous action space
        if self.config.node_prior == "uniform":  # 所有子节点的先验奖励相同，按均匀分布计算。
            prior_score = pb_c * (1 / len(parent.children))
        elif self.config.node_prior == "density":  # 根据子节点的先验概率密度计算，奖励与子节点的先验概率成比例。
            prior_score = pb_c * (
                child.prior / sum([child.prior for child in parent.children.values()])
            )
        else:
            raise ValueError("{} is unknown prior option, choose uniform or density")
        # 如果子节点已被访问 (child.visit_count > 0)，则计算节点的值评分。
        # 值评分是节点的奖励和折扣后的值的和。如果有多个玩家，值取反。
        # min_max_stats.normalize 用于将值评分归一化。
        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0
        # 总 UCB 评分是先验奖励和值评分的和。
        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """ 在每次模拟结束时，将节点的评价向上回传至根节点，更新树中所有经过的节点的统计信息。
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            # 如果只有一个玩家，所有经过的节点的值总和 (value_sum) 增加当前的值 (value)。
            # 更新节点的访问次数 (visit_count)。
            # 使用折扣后的当前节点值更新 min_max_stats。
            # 更新 value 以便传递给父节点。
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            # 如果有两个玩家，根据当前节点的 to_play 属性决定值的符号，只有当前玩家的节点值才被累加。
            # 更新节点的访问次数 (visit_count)。
            # 更新 min_max_stats，使用折扣后的值进行调整。
            # 更新 value，根据 to_play 属性决定节点的奖励符号。
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior=None):
        self.visit_count = 0  # 记录节点被访问的次数。用于计算平均值和决定是否需要进一步扩展节点。
        self.to_play = -1   # 表示当前轮到哪个玩家。用于确定每个节点下一个行动的执行者。
        self.prior = prior  # 节点的先验概率，通常在节点扩展时初始化。表示该节点的初始选择概率。
        self.value_sum = 0  # 记录从该节点起的累计价值，用于计算该节点的平均值。
        self.children = {}  # 存储当前节点的所有子节点。键是采取的动作，值是对应的子节点。
        self.hidden_state = None  # 记录与节点状态相关的隐藏信息，通常由神经网络生成。
        self.reward = 0  # 记录从该节点获得的即时奖励。

        self.mu = None  # 节点的动作分布的均值，通常由神经网络提供。
        self.std = None  # 节点的动作分布的标准差，通常由神经网络提供。

    def expanded(self):  # 判断节点是否有子节点
        return len(self.children) > 0

    def value(self):
        # 计算节点的平均值。用累计值除以访问次数。
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, to_play, reward, mu, log_std, hidden_state):
        """ 扩展节点，使用神经网络提供的价值、奖励、策略预测等信息来创建子节点。
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        self.mu = mu
        self.std = torch.exp(log_std)

        distribution = torch.distributions.normal.Normal(self.mu, self.std)
        action_value = distribution.sample().squeeze(0).detach().cpu().numpy()

        self.children[Action(action_value)] = Node()

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """  在每次搜索开始时，为根节点的先验概率添加 Dirichlet 噪声，以鼓励探索新的动作。
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

    def __repr__(self):
        return "value: %s, visit: %d, prior: %s, mu: %s, std: %s" % (
            self.value(),
            self.visit_count,
            self.prior,
            self.mu,
            self.std,
        )


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_actions = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                numpy.array([child.visit_count for child in root.children.values()])
            )

            self.root_values.append(root.value())
            self.child_actions.append(
                numpy.array([action.value for action in root.children.keys()])
            )
        else:
            self.root_values.append(None)
            self.child_actions.append(None)

    def map_action_to_observation(self, observation, action, delta_t=0.2):
        # 提取当前速度信息
        vx_current = observation[:, 3]  # vx
        vy_current = observation[:, 4]  # vy

        # 提取动作信息
        angle, acceleration = action[0], action[1]

        # 计算速度增量
        vx_increment = acceleration * delta_t * np.cos(angle)
        vy_increment = acceleration * delta_t * np.sin(angle)

        # 更新速度
        vx_new = vx_current + vx_increment
        vy_new = vy_current + vy_increment

        # 更新观测空间中的速度信息
        observation[:, 3] = vx_new  # 更新 vx
        observation[:, 4] = vy_new  # 更新 vy

        return observation

    def get_stacked_observations(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.

        NOTE(kwong): This code is duplicated below. But we'll leave it here so
        that we don't introduce new bugs to the existing MuZero implementation.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        # logger.info(f"value----{type(stacked_observations)}----{stacked_observations.shape}")
        # logger.info(f"value2----{type(stacked_observations[0])}----{stacked_observations[0].shape}???")
        for past_observation_index in reversed(
                range(index - num_stacked_observations, index)
        ):
            # logger.info(f"value2.5{past_observation_index}")
            if 0 <= past_observation_index:
                # logger.info(
                #     f"value2.6{type(self.observation_history[past_observation_index])}--{self.observation_history[past_observation_index].shape}")
                # logger.info(
                #     f"value2.7{type(self.action_history[past_observation_index + 1])}--{self.action_history[past_observation_index + 1]}")
                obs = self.map_action_to_observation(observation=numpy.ones_like(stacked_observations[0]),
                                                     action=self.action_history[past_observation_index + 1])
                # logger.info(f"value2.71{type(obs)}--{obs.shape}")
                previous_observation = numpy.concatenate(
                    (self.observation_history[past_observation_index], [obs],)
                )
            else:
                # logger.info(f"value2.8{type(self.observation_history[index])}--{self.observation_history[index].shape}")

                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )
            # logger.info(f"value3----{type(previous_observation)}----{previous_observation.shape}???")
            # logger.info(f"value4----{type(stacked_observations)}----{stacked_observations.shape}???")
            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )
            # logger.info(f"value5----{type(stacked_observations)}----{stacked_observations.shape}???")
        # logger.info(f"value_end----{type(stacked_observations)}----{stacked_observations.shape}???")
        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Action:
    """Class that represent an action of a game."""

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value.tostring())

    def __eq__(self, other):
        return (self.value == other.value).all()

    def __gt__(self, other):
        return self.value[0] > other.value[0]

    def __repr__(self):
        return str(self.value)
