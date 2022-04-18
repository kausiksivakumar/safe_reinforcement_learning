from safe_rl import ppo_lagrangian
import gym, safety_gym
import pdb


ppo_lagrangian(
	env_fn = lambda : gym.make('Safexp-CarGoal2-v0'),
	ac_kwargs = dict(hidden_sizes=(64,64))
	)