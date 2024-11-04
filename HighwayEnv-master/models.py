import math
from abc import ABC, abstractmethod

import ray
import torch


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                config.action_space,
                config.log_std_clamp,
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_mu_policy_layers,
                config.fc_log_std_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                config.action_space,
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        log_std_clamp,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_mu_policy_layers,
        fc_log_std_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.log_std_clamp = log_std_clamp
        self.full_support_size = 2 * support_size + 1
        # 表示网络，对观测空间进行降维，方便模型提取关键信息
        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )
        # 动力网络的一部分，接收拼接的潜在状态和动作，并输出下一步的潜在状态 next_encoded_state。
        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        # 基于 next_encoded_state 预测当前动作的即时奖励 reward的动力网络。
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )
        # 输出动作的均值的策略网络
        self.prediction_policy_mu_network = torch.nn.DataParallel(
            mlp(
                encoding_size,
                fc_mu_policy_layers,
                self.action_space_size,
                # output_activation=torch.nn.Tanh,
            )
        )
        # 输出动作的标准差的对数的策略网络
        self.prediction_policy_logstd_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_log_std_policy_layers, self.action_space_size)
        )
        # 价值网络
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        """
        将 next_encoded_state 输入到价值网络和策略网络中：
        价值网络输出 value，表示下一状态的估计价值。
        策略网络输出 mu 和 log_std，用于定义下一状态的动作分布。
        :param encoded_state: 表示网络返回的潜在状态
        :return:
        """
        # 表示在当前状态下模型倾向选择的动作。
        mu = self.prediction_policy_mu_network(encoded_state)
        # 动作分布的标准差的对数。控制动作的随机性，允许模型在不确定的状态下输出更具探索性的动作。
        log_std = self.prediction_policy_logstd_network(encoded_state)
        log_std = torch.clamp(log_std, *self.log_std_clamp)
        # mu 和 log_std 被用于定义策略网络中的动作分布，MuZero 可以在不同情境下生成适应性强的动作，并灵活调整探索与利用的平衡。
        # value返回当前状态的估计价值
        value = self.prediction_value_network(encoded_state)
        return mu, log_std, value

    def representation(self, observation):
        """
        对观测空间进行降维，提取关键特征
        :param observation:
        :return:
        """
        rep_input = observation.view(observation.shape[0], -1)
        ray.logger.info(f"representation_network: {rep_input.shape}")
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        ray.logger.info(f"representation_network size{encoded_state.shape}")
        # Scale encoded state between [-1, 1]
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        mean_encoded_state = (max_encoded_state + min_encoded_state) / 2
        scale_encoded_state = (max_encoded_state - min_encoded_state) / 2
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - mean_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        """
        输入当前的 encoded_state 和 action，预测下一潜在状态 next_encoded_state
        和奖励 reward。
        :param encoded_state: 表示网络生成的潜在状态的向量，表示当前环境的内部特征
        :param action: 当前采取的动作，通常是one-hot 编码的动作向量
        :return:
        """
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        # 将当前的潜在状态 encoded_state 与动作 action 沿着特征维度拼接（dim=1），生成一个包含状态和动作信息的输入 x。
        # 这种拼接操作可以帮助网络理解在当前状态下采取某一动作的效果
        x = torch.cat((encoded_state, action), dim=1)
        # 接收拼接后的 x，并输出下一步的潜在状态 next_encoded_state。
        next_encoded_state = self.dynamics_encoded_state_network(x)
        # 基于 next_encoded_state 预测当前动作的即时奖励 reward。
        reward = self.dynamics_reward_network(next_encoded_state)
        # 对 next_encoded_state 进行标准化，以确保输出的潜在状态在一定的数值范围内，
        # 这有助于提高模型的数值稳定性。
        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        mean_next_encoded_state = (max_next_encoded_state + min_next_encoded_state) / 2
        scale_next_encoded_state = (max_next_encoded_state - min_next_encoded_state) / 2
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - mean_next_encoded_state
        ) / scale_next_encoded_state
        # 返回下一潜在状态和奖励
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        """
        初始推理
        :param observation:
        :return:
        """
        # 通过表示网络 representation 将当前状态的观测空间转换为潜在表示 encoded_state。
        encoded_state = self.representation(observation)
        # 调用 prediction，通过 encoded_state 输入到价值网络和策略网络中：
        # 价值网络输出 value，表示初始状态的估计价值。
        # 策略网络输出 mu 和 log_std，定义初始状态下的动作分布。
        mu, log_std, value = self.prediction(encoded_state)
        # 通过固定逻辑生成一个奖励
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            mu,
            log_std,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        """
        递归推理
        :param encoded_state:
        :param action:
        :return:
        """
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        mu, log_std, value = self.prediction(next_encoded_state)
        return value, reward, mu, log_std, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        action_dim,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - action_dim)
        self.bn = torch.nn.BatchNorm2d(num_channels - action_dim)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - action_dim) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(
            num_channels - action_dim, reduced_channels_reward, 1
        )
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward,
            fc_reward_layers,
            full_support_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        x = self.conv1x1_reward(x)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.mu_fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )
        self.log_std_fc_policy = mlp(
            self.block_output_size_policy,
            [],
            action_space_size,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        mu = self.mu_fc_policy(policy)
        log_std = self.log_std_fc_policy(policy)
        return mu, log_std, value


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
        downsample,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        self.representation_network = torch.nn.DataParallel(
            RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        self.dynamics_network = torch.nn.DataParallel(
            DynamicsNetwork(
                num_blocks,
                num_channels + self.action_space_size,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
                self.action_space_size,
            )
        )

        self.prediction_network = torch.nn.DataParallel(
            PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )
        )

    def prediction(self, encoded_state):
        mu, log_std, value = self.prediction_network(encoded_state)
        return mu, log_std, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    action.shape[1],
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = action[:, :, None, None] * action_one_hot
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        mu, log_std, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        return (
            value,
            reward,
            mu,
            log_std,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        mu, log_std, value = self.prediction(next_encoded_state)
        return value, reward, mu, log_std, next_encoded_state


########### End ResNet ###########
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
