# Safe Reinforcement Learning
CAP adaptation for safety_gym environment

Environment: Safexp-PointGoal1-v0
#Installation
1) Create conda virtual encironment with python 3.6 (Only use python 3.6.13, mujoco_py installation fails otherwise)
2) Copy mujoco210 folder
3) `pip install -U 'mujoco-py<2.2,>=2.1'`
4) `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`
5) `sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so`
6) CONDA INSTALL FOR EVERYTHING ELSE - numpy, clff, patchelf, imageio
7) `sudo apt-get install libglew-dev`
8) `sudo apt-get install --reinstall libglvnd0`
9) Install `safety_gym` from https://github.com/openai/safety-gym and Mujoco by following the steps
10) Edit or comment `mujoco` in the seup.py under `/safety-gym` and add `mujoco-py<2.2,>=2.1` instead
11) Install starter rl kit from https://github.com/openai/safety-starter-agents and do the same as 3 for this environment as well



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

# To install environment

`conda env create -f environment.yml`

## To add the environment packages in your own conda environment 

`conda env update --name <name> --file environment.yml --prune`

# If running CAP without CUDA
```
python cap-planet/run_cap_planet.py --env Safexp-PointGoal1-v0 --cost-limit 0 --binary-cost --cost-constrained --penalize-uncertainty --learn-kappa --penalty-kappa 0.1 --disable-cuda --symbolic-env
```
# If running CAP with CUDA

python cap-planet/run_cap_planet.py --env Safexp-PointGoal1-v0 --cost-limit 0 --binary-cost --cost-constrained --penalize-uncertainty --learn-kappa --penalty-kappa 0.1 --symbolic-env

# To run CAP
1) Modify the setup file (i.e) comment the `mujoco-py==2.0.2.7` requirement since we already have `mujoco-py<2.2,>=2.1`
2) Install requirements using the `environment.yml` with steps provided as above
3) Install mujoco200 from https://www.roboti.us/download.html
### TODO Add encoding at top of run_cap_planet

# Running on EC2
1) Attach to session with given pem file (ssh) 
2) Run `tmux attach-session -t CAP_original` This is where the CAP code is currently running
3) `ctrl +B` and `D` to detach from the session (this does not mean that the tmux session is stopped)
4) Use `tmux kill-session -t session_name` to stop the session
5) ## Sessions running right now `CAP_original` : This is running the CAP-planet code
`python cap-planet/run_cap_planet.py --env Safexp-PointGoal1-v0 --binary-cost --cost-limit 0 --state-size 60 --belief-size 60 --hidden-size 60 --cost-constrained --penalize-uncertainty --learn-kappa --penalty-kappa 0.1 --symbolic-env --max-episode-length 1000 --episodes 1000 --planning-horizon 50 --checkpoint-experience --cost-discount 1`
Choosing Binary Cost(0,1) hence cost limit(C) =0 makes sense. State size is set as 60 because of gym observation size. action Repeat happening by default

