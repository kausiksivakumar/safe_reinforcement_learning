# CAP-PG: CONSERVATIVE ADAPTIVE PENALTY WITH POLICY GRADIENTS FOR MODEL BASED SAFE RL
### Topic: Safe Reinforcement Learning
### ESE650 Final Project Repository
This repository contains the code/results corresponding to our ESE650 Final Project on Safe Reinforcement Learning. 

Different branches of the repository contain code/results as indicated below:
1. CAP adaption for safety-gym environment (branch: CAP_original)
2. Our implemnetation of CAP-PG(very Slow) (branch: CAP_CCEM)
3. Our implemnetation of CAP-PG(Faster) (branch: CAP_CCEM_FAST)

## Maintainers
For all questions and issues please reach to Swati Gupta(gswati@seas.upenn.edu), Jasleen Dhanoa(jkdhanoa@seas.upenn.edu) and Kausik Sivakumar(kausik@seas.upenn.edu). 

## Contributions
- Kausik: CAP, CAP_PG, CAP_PG_FAST, Transition model, Running on AWS, Hyperparameter tuning, Installation and building environment
- Swati: CAP, CAP_PG, CAP_PG_FAST, Cost model, Running on AWS, Hyperparameter tuning, Installation and building environment
- Jasleen: Baselines, CAP, CAP_PG, CAP_PG_FAST, Reward model, Running on AWS, Hyperparameter tuning, Installation and building environment

# Installations needed for running this repository
## Mujoco(This is the trickiest step!)
1) Create conda virtual encironment with python 3.6 (Use only python 3.6.13, mujoco_py installation fails otherwise)
2) Install mujoco_py: 'pip3 install -U 'mujoco-py<2.2,>=2.1'
3) Run `import mujoco_py` in python terminal. If there is no error proceed to check if your installation is done properly(step 10)
4) If you face GLEW initalization error(especially on AWS): do steps 4-9
5) `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`
6) `sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so`
7) `sudo apt-get install libglew-dev`
8) `sudo apt-get install --reinstall libglvnd0`
9) Do CONDA INSTALL FOR EVERYTHING ELSE which might be missing while running `import mujoco_py` in python terminal e.g.- numpy, clff, patchelf, imageio 
10) Check if your installation of mujoco_py is done properly by making an environment: https://github.com/openai/mujoco-py
## Gym Installation
1) Install gym via pip if it is not already installed: `pip install gym`
## Safety Gym Installation
1) Install `safety_gym`by following steps 1-3: `git clone https://github.com/openai/safety-gym.git`
2) `cd safety-gym`
3) `pip install -e .`
4) Edit or comment `mujoco` in the seup.py under `/safety-gym` and add `mujoco-py<2.2,>=2.1` instead as we have mujoco210
5) Check if the installation is done properly by making an environment:https://github.com/openai/safety-gym(Mujoco needs to be installed prior to this!)
## Torch 
Pytorch needs to be installed and requires the appropriate version of CUDA if running on GPU
## CAP Installation
1) In order to modify your existing environment to be able to run our adaptation of CAP code for safety-gym use the `environment.yml` file provided in the repository e.g. `conda env update --name <name> --file environment.yml --prune`
2) Modify the setup file (i.e) comment the `mujoco-py==2.0.2.7` requirement since we already have `mujoco-py<2.2,>=2.1`
3) Install mujoco200(mujoco200_linux was used by us) from https://www.roboti.us/download.html alongside mujoco210(already installed)
4) Run CAP follwing commands given below:
## Running on EC2 instance(optional)
*The EC2 instance needs all of the above installations.*

0) Create an EC2 instance and ssh into it using pem file 
1) Create a new tmux session- `tmux new -s [name]`
2) Run the CAP code by specifying the hyperparameters as shown below.
3) Detach and attach from the session as required.
4) It will generate live train and test plots for you to monitor the progress of your model

# Evaluating Baselines (from safety-starter-pack)
1) Install starter RL kit from https://github.com/openai/safety-starter-agents for getting the Baselines code
2) The following Robots, Tasks and RL-Algorithms are available for training and plotting in the safety-starter-agents code:
- robot_list = ['point', 'car', 'doggo']
- task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
- algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']
3) Choose the robot, task and algo to create to train on a suitable environment. In our paper, we have run all experiments on point goal1 environment for all of the above algorithms. e.g. `time python experiment.py --algo cpo --task goal1 --robot point --seed 20 --cpu 8`
4) In order to perform plotting use the plot.py script provided in safety-starter-agents code. eg:
```
python plot.py 2022-04-15_ppo_PointGoal1/2022-04-15_21-45-36-ppo_PointGoal1_s0 2022-04-16_trpo_PointGoal1/2022-04-16_00-25-36-trpo_PointGoal1_s0 ppo_lagrangian_PointGoal1/ 2022-04-14_20-51-30-trpo_lagrangian_PointGoal1_s20/ 2022-04-15_01-23-25-cpo_PointGoal1_s20/ --legend 'PPO' 'TRPO' 'PPO Lagrangian' 'TRPO Lagrangian' 'CPO'  --value 'CostRate' 'Entropy' 'KL' 'AverageEpRet' 'AverageEpCost' --smooth 10 
```

# Running CAP (our adaptation for safety-gym, our PG implementations)
Please go to the appropriate branch to run the CAP code as given above. The commands to run the code are the same.
#### If running CAP without CUDA
```
python3 cap-planet/run_cap_planet.py --env Safexp-PointGoal1-v0 --cost-limit 0 --binary-cost --cost-constrained --penalize-uncertainty --learn-kappa --penalty-kappa 0.1 --disable-cuda --symbolic-env
```
#### If running CAP with CUDA
```
python3 cap-planet/run_cap_planet.py --env Safexp-PointGoal1-v0 --cost-limit 0 --binary-cost --cost-constrained --penalize-uncertainty --learn-kappa --penalty-kappa 0.1 --symbolic-env
```

<<<<<<< HEAD
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
`python3 cap-planet/run_cap_planet.py --env Safexp-PointGoal1-v0 --binary-cost --cost-limit 0 --state-size 60 --belief-size 60 --hidden-size 60 --cost-constrained --penalize-uncertainty --learn-kappa --penalty-kappa 0.1 --symbolic-env --max-episode-length 1000 --episodes 1000 --planning-horizon 50 --checkpoint-experience --cost-discount 1`
Choosing Binary Cost(0,1) hence cost limit(C) =0 makes sense. State size is set as 60 because of gym observation size. action Repeat happening by default

=======
# Useful Repositories
We acknowledge and thank the efforts of the maintainers of these wondeful repositories. It wouldn't have been possible to do this project without building up on top of their efforts!
- Github repo for the original CAP implementation: https://github.com/Redrew/CAP
- Github repo for safety-starter agents: https://github.com/openai/safety-starter-agents
- Github repo mujoco_py:https://github.com/openai/mujoco-py
>>>>>>> 5b6e07d... README update
