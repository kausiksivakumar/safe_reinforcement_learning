# safe_reinforcement_learning
CAP adaptation for safety_gym environment

#Installation
1) Create conda virtual encironment with python 3.6
2) Install `safety_gym` from https://github.com/openai/safety-gym and Mujoco by following the steps
3) Edit or comment `mujoco` in the seup.py under `/safety-gym` and add `mujoco-py<2.2,>=2.1` instead
4) Install starter rl kit from https://github.com/openai/safety-starter-agents and do the same as 3 for this environment as well



# Evaluating Baseline (from safety-starter-pack)
task: goal1, robot: point, algos: all

robot_list = ['point', 'car', 'doggo']
task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
    
Ex:
`time python experiment.py --algo cpo --task goal1 --robot point --seed 20`

 Add `--cpu` to use more than one core Ex `--cpu 4` to use 4 cores
 
 # Dimensions of gym
 action_space - `Box(2,) [-1,1]`
 observation_space - `Box(60,) [-inf,inf]`

# Plotting
In order to perform plotting use the plot.py script.
Example:
```
python plot.py 2022-04-15_ppo_PointGoal1/2022-04-15_21-45-36-ppo_PointGoal1_s0 2022-04-16_trpo_PointGoal1/2022-04-16_00-25-36-trpo_PointGoal1_s0 ppo_lagrangian_PointGoal1/ 2022-04-14_20-51-30-trpo_lagrangian_PointGoal1_s20/ 2022-04-15_01-23-25-cpo_PointGoal1_s20/ --legend 'PPO' 'TRPO' 'PPO Lagrangian' 'TRPO Lagrangian' 'CPO'  --value 'CostRate' 'Entropy' 'KL' 'AverageEpRet' 'AverageEpCost' --smooth 10 
```