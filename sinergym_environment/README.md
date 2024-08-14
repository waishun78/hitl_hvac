# Sinergym Overview

Refer to [GUIDE.md](GUIDE.md) for installation instructions.

### Files and Their Contents

#### `__init__.py`

- **Purpose**: Sets up and registers different environments in Sinergym.
- **Key Components**:
  - Environment registration using `gymnasium` with parameters such as action space, variables, actuators, and reward functions.
  - Reads environment configuration files and registers environments accordingly.
- **What You Can Change**:
  - Modify the parameters like action space, observation variables, and reward functions to customize environments.
  - Add new environment configurations by including new files in the configuration directory.

#### `eplus_env.py`

- **Purpose**: Defines the main environment class `EplusEnv` for simulation with EnergyPlus.
- **Key Components**:
  - Constructor initializes the environment with building and weather files, action space, variables, meters, actuators, and reward functions.
  - Methods for resetting the environment, stepping through actions, and handling observations and rewards.
- **What You Can Change**:
  - Update `action_space` and `observation_space` to define the actions and observations in your environment.
  - Customize the `reward` function to change how the agent is rewarded for its actions.
  - Modify `variables`, `meters`, and `actuators` to add or change the elements being controlled and monitored.

#### `wrapper.py`

- **Purpose**: Provides custom wrappers to enhance or modify the behavior of the environment.
- **Key Components**:
  - `NormalizeObservation`: Normalizes observations to improve stability in training.
  - `NormalizeAction`: Normalizes actions to a specific range.
  - `LoggerWrapper`: Logs interactions with the environment for analysis.
  - Other wrappers for multi-objective rewards, stacking observations, and modifying action or observation spaces.
- **What You Can Change**:
  - Use or modify wrappers to adjust how observations and actions are processed.
  - Implement new wrappers if you need custom behavior not provided by existing ones.
  - Adjust logging settings in `LoggerWrapper` to customize what information is recorded during simulations.

#### `try_env.py`

- **Purpose**: A sample script to demonstrate how to set up and run a Sinergym environment.
- **Key Components**:
  - Imports necessary libraries and sets environment variables for EnergyPlus paths.
  - Creates an environment and applies normalization and logging wrappers.
  - Runs a simulation for a specified number of episodes, executing random actions and printing rewards.
- **What You Can Change**:
  - Adjust the number of episodes or actions taken during the simulation.
  - Modify the environment settings, such as which environment to create or which wrappers to apply.

#### `rewards.py`

- **Purpose**: Contains implementations of reward functions used to evaluate the agent's performance.
- **Key Components**:
  - `BaseReward`: A base class for creating custom reward functions.
  - `LinearReward`: Calculates rewards based on energy consumption and comfort levels.
  - `ExpReward`: Similar to `LinearReward` but uses an exponential function for temperature comfort differences.
  - `HourlyLinearReward`: A linear reward function with time-dependent weights.
  - `NormalizedLinearReward`: Normalizes reward terms for energy and comfort penalties.
- **What You Can Change**:
  - Implement custom reward functions by inheriting from `BaseReward`.
  - Modify existing reward functions to adjust how energy and comfort are weighted.
  - Add new reward functions to evaluate different performance criteria.

### How to Modify the Environment

1. **Change Action and Observation Spaces**: Update the `action_space` and `observation_space` in `eplus_env.py` to control what actions the agent can take and what observations it receives.

2. **Customize Rewards**: In `eplus_env.py`, adjust the `reward` function to reflect your desired objectives, such as minimizing energy consumption or maximizing comfort.

3. **Add or Modify Variables and Actuators**: Use the `variables`, `meters`, and `actuators` definitions in `eplus_env.py` to monitor different aspects of the building or control new components.

4. **Use Wrappers**: Apply existing wrappers in `wrapper.py` to enhance the environment or create new ones for specific modifications.

5. **Test with `try_env.py`**: Use the `try_env.py` script to quickly test changes to the environment and observe the effects on simulation performance.
