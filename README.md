# safe_reinforcement_learning
CAP adaptation for safety_gym environment

#Installation
1) Create conda virtual encironment with python 3.6
2) Install `safety_gym` from https://github.com/openai/safety-gym and Mujoco by following the steps
3) Edit or comment `mujoco` in the seup.py under `/safety-gym` and add `mujoco-py<2.2,>=2.1` instead
4) Install starter rl kit from https://github.com/openai/safety-starter-agents and do the same as 3 for this environment as well



# Evaluating Baseline (from safety-starter-pack 
task: goal1, robot: point, algos: all

robot_list = ['point', 'car', 'doggo']
task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
    
Ex:
`time python experiment.py --algo cpo --task goal1 --robot point --seed 20`

 Add `--cpu` to use more than one core Ex `--cpu 4` to use 4 cores
