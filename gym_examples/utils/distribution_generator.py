# import numpy as np
# import pandas as pd


# def generate_hourly_pop_distribution(met_mu:int, met_sigma:int, clo_mu:int, clo_sigma:int, n_enter_mu:int, n_enter_sigma:int, n_exit_mu:int, n_exit_sigma:int, directory:str) -> None:
#     """
#     Generate a csv file of hourly metabolic (0.8-4 limit) and clothing insulation (0-2 limit) based on the ISO standards https://pythermalcomfort.readthedocs.io/en/latest/reference/pythermalcomfort.html.
#     The file will be saved in the specified directory.
    
#     Parameters:
#     met_mu (int): Mean metabolic rate
#     met_sigma (int): Standard deviation of metabolic rate
#     clo_mu (int): Mean clothing insulation
#     clo_sigma (int): Standard deviation of clothing insulation
#     n_enter_mu (int): Mean number of people entering the system
#     n_enter_sigma (int): Standard deviation number of people entering the system
#     n_exit_mu (int): Mean number of people exiting the system
#     n_exit_sigma (int): Standard deviation number of people exiting the system
    
#     directory (str): Directory where the CSV file will be saved
    
#     The CSV file will be of the following format:
#     hour, met_mu, met_sigma, clo_mu, clo_sigma, num_mu, num_sigma
#     0, metm_0, mets_0, clom_0, clos_0, numm_0, nums_0
#     1, metm_1, mets_1, clom_1, clos_1, numm_1, nums_1
#     .
#     .
#     .
#     23, met_23, clo_23
#     """
#     # Note: Fill is used as there, mu and sigma within each episode is kept the same (actualised results provide different distributions already)
#     hours = np.arange(24)

#     # Generating the hourly met mu, sigma
#     hourly_met_mu = np.fill((24,), met_mu)
#     hourly_met_sigma = np.fill((24,), met_sigma)

#     # Generating the hourly clo mu, sigma
#     hourly_clo_mu = np.fill((24,), clo_mu)
#     hourly_clo_sigma = np.fill((24,), clo_sigma)

#     # Generating the n_enter mu, sigma
#     hourly_enter_mu = np.fill((24,), clo_mu)
#     hourly_enter_sigma = np.fill((24,), clo_sigma)

#     # Generating the exit mu, sigma
#     hourly_exit_mu = np.fill((24,), clo_mu)
#     hourly_exit_sigma = np.fill((24,), clo_sigma)

#     data = {
#         'hour': hours,
#         'met': met_values,
#         'clo': clo_values
#     }
#     df = pd.DataFrame(data)

