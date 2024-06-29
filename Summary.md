### Files added their purpose

**aircon_model.ipynb**

- Training the two models (hitl+energy and pure energy)
- Code to run it in a continuous loop and not episodic. It saves after every 50 episdodes and uses it as a starting point when run again

**ensembling.ipynb**

- This file uses ensemble to improve model performance
- The two pretrained models are used and their outputs are combined using majority voting or boltzmann addition
- The combined model selects actions based on the ensemble strategy and the model is trained

**comparison.ipynb**

- The cumulative rewards for each model are collected and plotted for comparison
- The average cumulative rewards are calculated to quantify the performance of each model
