from dm_control import suite
import gymnasium as gym

env_gym = gym.make("CartPole-v1")
env = suite.load("walker", "walk")

print(env.action_spec())
print(env.observation_spec())
print(env_gym.action_space)