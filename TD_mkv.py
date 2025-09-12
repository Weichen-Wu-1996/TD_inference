import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import yaml
import click
import os
from MRP import Markov_Reward_Process

def TD_mkv_one_trial(MRP: Markov_Reward_Process,
                     save_iter: list,
                     initial_stepsize: float,
                     alpha: float,
                     theta0: np.array = None,
                     save_original: bool = False,
                     initial_distribution = 'uniform'):
    '''
    Function to run the averaged TD learning algorithm
    with polynomial-decay stepsizes and iid samples
    MRP: The Markov reward process
    save_iter: list of iterations to save
    initial_stepsize: the initial stepsize 
    alpha: the polynomial decay parameter, must be in [0.5,1]
    theta0: the initial iteration, default to 0
    save_original: whether to save the original TD iterates
    initial_distribution: the distribution of the initial state s0,
                          can be "uniform", "stationary",
                          or any number in [0,S)
    seed: random seed to guarantee reproduction
    '''
    assert alpha >= 0.5 and alpha <= 1, "alpha must be within [0.5,1]!"

    # Suppress divide by zero, overflow, and invalid warnings globally
    np.seterr(divide='ignore', over='ignore', invalid='ignore')
    
    
    
    # Total number of iterations
    T = max(save_iter) + 1
    n_save = len(save_iter)

    # Initialize estimator
    if not theta0:
        theta = np.zeros((MRP.d,))
    else:
        theta = theta0
    theta_bar = theta.copy()
    saved_theta_bars = np.zeros((n_save, MRP.d))
    theta_star = MRP.theta_star[:,0]

    # Initial state
    if initial_distribution == 'uniform':
        p = np.ones((MRP.S)) / MRP.S
    elif initial_distribution == 'stationary':
        p = MRP.mu
    else:
        p = np.zeros((MRP.S))
        p[initial_distribution] = 1
    s0 = np.random.choice(MRP.S, p = p)

    if save_original:
        saved_thetas = np.zeros((n_save, MRP.d))

    for t in range(1,T):
        s1 = np.random.choice(MRP.S, p = MRP.P[s0])
        TD_err = - MRP.r[s0] + np.sum((MRP.Phi[s0] - MRP.gamma * MRP.Phi[s1]) * theta)
        stepsize = initial_stepsize * t ** (-alpha)
        delta = theta - theta_star
        theta -= stepsize * TD_err * MRP.Phi[s0]
        theta_bar += (theta - theta_bar) / t
        s0 = s1

        if t in save_iter:
            i = save_iter.index(t)
            saved_theta_bars[i] = theta_bar

            if save_original:
                saved_thetas[i] = theta

    one_trial_result = {'saved_delta_bars': saved_theta_bars - theta_star[None,:]}
    if save_original:
        one_trial_result['saved_deltas'] = saved_thetas - theta_star[None,:]

    return one_trial_result

def merge_trial_results(one_trial_results):
    results = {}
    for key in one_trial_results[0].keys():
        results[key] = np.array([result[key] for result in one_trial_results])
        results[key] = np.swapaxes(results[key], 0, 1)
    return results

def TD_mkv_multi_trials(MRP: Markov_Reward_Process,
                        save_iter: list,
                        N_trials: int,
                        initial_stepsize: float,
                        alpha: float,
                        theta0: np.array = None,
                        initial_distribution = 'stationary',
                        save_original: bool = False,
                        seed: int = 42):

    # set random seed
    random.seed(seed)
    
    one_trial_results = Parallel(n_jobs=-1)(delayed(TD_mkv_one_trial)(MRP,
                                                                      save_iter,
                                                                      initial_stepsize,
                                                                      alpha,
                                                                      theta0,
                                                                      save_original,
                                                                      initial_distribution) for _ in tqdm(range(N_trials)))
    results = merge_trial_results(one_trial_results)
    results.update({'T': max(save_iter) + 1,
                    'save_iter': save_iter,
                    'initial_stepsize': initial_stepsize,
                    'alpha': alpha})
    return results

@click.command()
@click.argument("data-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(data_dir):
    # Load config
    with open(os.path.join(data_dir,"config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Construct Markov Reward Process
    MRP = Markov_Reward_Process(**config["MRP"])

    # Conduct experiment
    mkv_results = TD_mkv_multi_trials(MRP, **config["mkv_experiment"])

    # Save data and results
    MRP_np = {k: np.array(v) for k, v in MRP.__dict__.items()}
    np.savez(os.path.join(data_dir,"MRP.npz"), **MRP_np)
    print("MRP data saved at", os.path.join(data_dir,"MRP.npz"))

    mkv_results = {k: np.array(v) for k, v in mkv_results.items()}
    np.savez(os.path.join(data_dir,"mkv_results"), **mkv_results)
    print("Experiment data saved at", os.path.join(data_dir,"mkv_results"))
    

if __name__ == "__main__":
    main()


            
    

    