import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def compare_histograms(MRP_params,
                       results,
                       iter_index,
                       xlabel):
    emp_hist = plt.hist(results['saved_delta_bars'][iter_index,:,0] * np.sqrt(results['save_iter'][iter_index]), 
                        density = True, 
                        bins = 50, 
                        range = (-1,1), 
                        label = 'Empirical')
    pts = np.linspace(-1, 1, 1000)
    asy_hist = norm.pdf(pts, scale = np.sqrt(MRP_params['Lambda_star'][0,0]))
    plt.plot(pts,asy_hist,'-',color = 'red', label = 'Asymptotic')
    plt.xlim(-1,1)
    plt.xlabel(xlabel)
    plt.ylabel('Probability density')
    plt.legend()

def one_trial_CIs(MRP_params, 
                  results, 
                  sample_index = 3,
                  confidence_level = 0.95, 
                  xmin = 1024,
                  yerr_lim = 0.01):
    
    plt.rcParams['text.usetex'] = True
    fig, axs = plt.subplots(1, 3, figsize = (12,4), dpi = 300)

    cv = norm.ppf((1 + confidence_level) / 2)
        
    ts = results['save_iter']
    saved_theta_bars = results['saved_delta_bars'] + MRP_params['theta_star'].reshape((1,1,-1))
    saved_Lambda_hats = results['saved_Lambda_hats']
    theta_star = MRP_params['theta_star']
    
    for dim in range(3):
        axs[dim].plot(ts, saved_theta_bars[:,sample_index,dim], color = 'royalblue')
        axs[dim].fill_between(ts,
                             saved_theta_bars[:,sample_index,dim] - cv * np.sqrt(saved_Lambda_hats[:,sample_index,dim,dim]/ts),
                             saved_theta_bars[:,sample_index,dim] + cv * np.sqrt(saved_Lambda_hats[:,sample_index,dim,dim]/ts),
                             color = 'lightblue', alpha = 0.5)
        axs[dim].plot(ts, np.tile(theta_star[dim],len(ts)),'-.', color = 'black')
        axs[dim].set_ylim(theta_star[dim] - yerr_lim,theta_star[dim] + yerr_lim)
        axs[dim].set_xlim(xmin, ts[-1])
        axs[dim].set_xlabel(r'$T$')
        axs[dim].set_ylabel(r'$\theta_' + str(dim+1) + '$')
        axs[dim].set_xscale('log')
           
    plt.subplots_adjust(wspace = 0.4)

def estimate_norm_quantiles(V, q, nsamples = 10 ** 6, seed = 42):
    d = V.shape[0]
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mean=np.zeros(d), cov=V, size=nsamples)
    L2_norms = np.sum(X ** 2, axis = 1)
    return np.quantile(L2_norms,q)

def plot_L2_norm_quantiles(results, q, label):
    L2_errs = np.sum(results['saved_delta_bars'] ** 2, axis = 2)
    L2_quantiles = np.quantile(L2_errs, q, axis = 1)
    plt.plot(results['save_iter'],L2_quantiles, label = label)

def compare_emp_asy(MRP_params, results, label):
    n_saved, N_trials, d = results['saved_delta_bars'].shape
    reg = results['save_iter'][:,np.newaxis] / np.diagonal(MRP_params['Lambda_star'])[np.newaxis,:]
    sorted_delta_bars = np.sort(results['saved_delta_bars'], axis = 1) * np.sqrt(reg)[:,np.newaxis,:]
    emp_cdf = (np.arange(1,N_trials+1) / N_trials)[np.newaxis,:,np.newaxis]
    est_cdf = norm.cdf(sorted_delta_bars)
    dists = np.max(abs(emp_cdf - est_cdf), axis = (1,2))
    plt.plot(results['save_iter'], dists, label = label)

def plot_variance_estimation_errors(MRP_params, results, label):
    F_errs = np.sum((results['saved_Lambda_hats'] - MRP_params['Lambda_star'][np.newaxis,np.newaxis,:,:])** 2, axis = (2,3))
    plt.plot(results['save_iter'],np.mean(F_errs, axis = 1), label = label)
