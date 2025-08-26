import numpy as np
import random
from tqdm import tqdm
import json
import yaml
import click
import os
from MRP import Markov_Reward_Process

def TD_iid(MRP: Markov_Reward_Process,
           save_iter: list,
           N_trials: int,
           initial_stepsize: float,
           alpha: float,
           theta0: np.array = None,
           estimate_variance: bool = True,
           save_original: bool = False,
           seed: int = 42):
    '''
    Function to run the averaged TD learning algorithm
    with polynomial-decay stepsizes and iid samples
    MRP: The Markov reward process
    save_iter: list of iterations to save
    N_trials: number of trials
    initial_stepsize: the initial stepsize 
    alpha: the polynomial decay parameter, must be in [0.5,1]
    theta0: the initial iteration, default to 0
    estimate_variance: whether to include variance estimator
    save_original: whether to save the original TD iterations
    seed: random seed to guarantee reproduction
    '''
    assert alpha >= 0.5 and alpha <= 1, "alpha must be within [0.5,1]!"
    
    # set random seed
    random.seed(seed)
    
    # Total number of iterations
    T = max(save_iter) + 1
    n_save = len(save_iter)

    # Initialize estimator
    if not theta0:
        thetas = np.zeros((N_trials,MRP.d))
    else:
        thetas = np.tile(theta0, (N_trials,1))
    theta_bars = thetas
    saved_theta_bars = np.zeros((n_save, N_trials, MRP.d))

    if save_original:
        saved_thetas = np.zeros((n_save, N_trials, MRP.d))
    
    if estimate_variance:
        A_bars = np.zeros((N_trials, MRP.d, MRP.d))
        AA_bars = np.zeros((N_trials, MRP.d ** 2, MRP.d ** 2))
        Ab_bars = np.zeros((N_trials, MRP.d ** 2, MRP.d))
        bb_bars = np.zeros((N_trials, MRP.d ** 2))

        saved_A_bars = np.zeros((n_save, N_trials, MRP.d, MRP.d))
        saved_AA_bars = np.zeros((n_save, N_trials, MRP.d ** 2, MRP.d ** 2))
        saved_Ab_bars = np.zeros((n_save, N_trials, MRP.d ** 2, MRP.d))
        saved_bb_bars = np.zeros((n_save, N_trials, MRP.d ** 2))

    # vectorize the stationary distribution of (s,s')
    muP = np.reshape(MRP.mu.reshape((MRP.S,1)) * MRP.P,(-1,))

    for t in tqdm(range(1,T)):
        # sampling
        samples = np.random.choice(MRP.S ** 2, p = muP, size = N_trials)
        s1 = samples // MRP.S
        s2 = samples % MRP.S
        
        # TD error
        TD_err = MRP.r[s1] - np.sum((MRP.Phi[s1] - MRP.gamma * MRP.Phi[s2]) * thetas, axis = 1)
        TD_err = TD_err.reshape((-1,1))
        
        # update theta
        thetas = thetas + initial_stepsize * (t ** (-alpha)) * TD_err * MRP.Phi[s1]
        theta_bars = theta_bars + (thetas - theta_bars)/ t 

        if estimate_variance:
            # update A, AA, Ab, bb
            Ats = (MRP.Phi[s1])[:,:,np.newaxis] * (MRP.Phi[s1] - MRP.gamma * MRP.Phi[s2])[:,np.newaxis,:]
            bts = MRP.r[s1][:,np.newaxis] * MRP.Phi[s1]
            A_bars += (Ats - A_bars) / t
            
            Ats_exp = Ats[:, :, np.newaxis, :, np.newaxis]
            AAs = (Ats_exp * Ats_exp.transpose(0, 2, 1, 4, 3)).reshape(N_trials, MRP.d ** 2, MRP.d ** 2)
            AA_bars += (AAs - AA_bars) / t
            
            Ats_exp = Ats[:,:,np.newaxis,:]
            bts_exp = bts[:,np.newaxis,:,np.newaxis]
            Abs = (Ats_exp * bts_exp).reshape(N_trials, MRP.d ** 2, MRP.d)
            Ab_bars += (Abs - Ab_bars) / t
            
            bts_exp = bts[:,:,np.newaxis]
            bbs = (bts_exp * bts_exp.transpose(0,2,1)).reshape(N_trials, MRP.d ** 2)
            bb_bars += (bbs - bb_bars) / t

        if t in save_iter:
            i = save_iter.index(t)
            saved_theta_bars[i] = theta_bars 

            if save_original:
                saved_thetas[i] = thetas

            if estimate_variance:
                saved_A_bars[i] = A_bars
                saved_AA_bars[i] = AA_bars
                saved_Ab_bars[i] = Ab_bars
                saved_bb_bars[i] = bb_bars            

    results = {'T': T,
               'save_iter': save_iter,
               'initial_stepsize': initial_stepsize,
               'alpha': alpha,
               'saved_theta_bars': saved_theta_bars}

    if save_original:
        results['saved_thetas'] = saved_thetas

    if estimate_variance:
        # generate variance estimators
        saved_Gamma_hats = np.zeros((n_save, N_trials, MRP.d, MRP.d))
        saved_Lambda_hats = np.zeros((n_save, N_trials, MRP.d, MRP.d))
        for i in range(n_save):
            for j in range(N_trials):
                theta_bar = saved_theta_bars[i][j]
                A_bar = saved_A_bars[i][j]
                AA_bar = saved_AA_bars[i][j]
                Ab_bar = saved_Ab_bars[i][j]
                bb_bar = saved_bb_bars[i][j]
                Gamma_hat = np.dot(AA_bar, np.kron(theta_bar,theta_bar)) - 2 * np.dot(Ab_bar,theta_bar) + bb_bar
                Gamma_hat = np.reshape(Gamma_hat,(MRP.d, MRP.d))
                saved_Gamma_hats[i][j] = Gamma_hat
                A_bar_inv = np.linalg.inv(A_bar + 1e-8 * np.identity(MRP.d))
                Lambda_hat = np.dot(np.dot(A_bar_inv,Gamma_hat),np.transpose(A_bar_inv))
                saved_Lambda_hats[i][j] = Lambda_hat
        results.update({'saved_A_bars': saved_A_bars,
                        'saved_Gamma_hats': saved_Gamma_hats,
                        'saved_Lambda_hats': saved_Lambda_hats})

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
    iid_results = TD_iid(MRP, **config["iid_experiment"])

    # Save data and results
    MRP_np = {k: np.array(v) for k, v in MRP.__dict__.items()}
    np.savez(os.path.join(data_dir,"MRP.npz"), **MRP_np)
    print("MRP data saved at", os.path.join(data_dir,"MRP.npz"))

    iid_results = {k: np.array(v) for k, v in iid_results.items()}
    np.savez(os.path.join(data_dir,"iid_results"), **iid_results)
    print("Experiment data saved at", os.path.join(data_dir,"iid_results"))
    

if __name__ == "__main__":
    main()

