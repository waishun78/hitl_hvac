### `change_environment.ipynb`

#### **Changing Rewards**
  ```python
  from sinergym.utils.rewards import ExpReward

  env = gym.make('Eplus-5zone-hot-continuous-v1', reward=ExpReward)
  ```
  - **What It Does** - Use a different reward function (ExpReward) defined in the utils folder.

#### **Updating Weather Files**
  ```python
  env = gym.make('Eplus-5zone-cool-continuous-stochastic-v1', weather_files='ESP_Granada.epw')
  ```
  - **What It Does** - Changes the weather data used in the simulation.

#### **Modifying Observation/Action Spaces**
  ```python
  new_time_variables = ['month', 'day_of_month', 'hour']
  env = gym.make('Eplus-5zone-hot-continuous-v1', time_variables=new_time_variables)
  ```
  - **What It Does** - Set custom observation variables and change what information the agent receives.

#### **Adding Extra Configuration**
  ```python
  extra_conf = {'timesteps_per_hour': 6, 'runperiod': (1, 1, 1991, 2, 1, 1991)}
  env = gym.make('Eplus-datacenter-cool-continuous-stochastic-v1', config_params=extra_conf)
  ```
  - **What It Does** - Changes how the simulation runs (adjusting timesteps per hour or the simulation period).

--- 

### `default_building_control.ipynb`

#### **Using Default Controls**
  ```python
  env = gym.make('Eplus-5zone-hot-continuous-v1')
  for i in range(1):
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        rewards.append(reward)

  ```
  - **What It Does** - Default sinergym environment setup
  
---

### `drl.ipynb`

#### **Training an RL Model**
  ```python
  from stable_baselines3 import PPO

  env = gym.make('Eplus-5zone-mixed-continuous-stochastic-v1')
  model = PPO('MlpPolicy', env, verbose=1)
  model.learn(total_timesteps=10000)
  ```
  - **What It Does** - Sets up and trains a PPO agent to interact with the Sinergym environment

#### **Using Environment Wrappers**
  ```python
  from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction

  env = NormalizeObservation(env)
  env = NormalizeAction(env)
  ```
  - **What It Does** - Applies normalization wrappers to standardize observations and actions (which is typically done in EnergyPlus simulations)

#### **Evaluating a Model**
  ```python
  env.reset()
  for _ in range(5):
      obs, rewards, done, _ = env.step(env.action_space.sample())
      total_reward += reward
      timesteps += 1
average_reward = total_reward / timesteps if timesteps > 0 else 0 print(f"Episode {episode + 1}: Cumulative Reward = {total_reward}, Average Reward = {average_reward}")

  ```
  - **What It Does** - Evaluate a trained agent over a number of episodes.
  
---

### `getting_env_information.ipynb`

#### **Exporting Schedulers to Excel**
  ```python
  export_schedulers_to_excel(schedulers=schedulers, path='./example.xlsx')
  ```
  - **What It Does** - Save the scheduler information of a building to an Excel file.

#### **Accessing Environment Properties**
  ```python
  env = gym.make('Eplus-demo-v1')
  print(env.get_wrapper_attr('observation_variables'))
  ```
  - **What It Does** - Prints the list of observation variables available in the environment.

#### **Displaying All Environment Information**
  ```python
  env.info()
  ```
  - **What It Does** - Comprehensive printout of the environment's current state,variables,  actions and configuration.
