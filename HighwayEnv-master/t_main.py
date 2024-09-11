import gymnasium as gym
import highway_env

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "order": "sorted",
        "normalize": True,
        },
    'action': {'type': 'ContinuousAction',
               'acceleration_range': (-4, 4.0)},
    'screen_width': 900,  # 屏幕宽度
    'screen_height': 600,  # 屏幕高度
    'render_agent': True,  # 控制渲染是否应用到屏幕
}
env = gym.make('highway-v0', render_mode="rgb_array", config=config)
obs, info = env.reset()
rgb_img = env.render()
