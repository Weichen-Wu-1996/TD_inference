import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter
from matplotlib.animation import FuncAnimation

def compare_histograms(MRP_params,
                       results,
                       iter_index,
                       xlabel,
                       dim = 0,
                       nbins = 50):
    sigma = np.sqrt(MRP_params['Lambda_star'][dim,dim])
    xlim = 5 * sigma
    emp_hist = plt.hist(results['saved_delta_bars'][iter_index,:,dim] * np.sqrt(results['save_iter'][iter_index]), 
                        density = True, 
                        bins = nbins, 
                        range = (-xlim,xlim), 
                        label = 'Empirical')
    pts = np.linspace(-xlim, xlim, 1000)
    asy_hist = norm.pdf(pts, scale = sigma)
    plt.plot(pts,asy_hist,'-',color = 'red', label = 'Asymptotic')
    plt.xlim(-xlim,xlim)
    plt.xlabel(xlabel)
    plt.ylabel('Probability density')
    plt.legend()

def animate_histograms(MRP_params,
                       results,
                       Tlabels,
                       dim,
                       fname,
                       nbins = 50):
    sigma = np.sqrt(MRP_params['Lambda_star'][dim,dim])
    xlim = 5 * sigma

    fig, ax = plt.subplots()
    bins = np.linspace(-xlim, xlim, nbins)

    # All bars set to 0
    counts = np.zeros(len(bins) - 1)
    _, _, patches = ax.hist([], bins=bins, 
                            alpha=0.7, 
                            color="steelblue", 
                            edgecolor="black",
                            label = "Empirical")
    for patch in patches:
        patch.set_height(0) 

    # Asymptotic distribution
    pts = np.linspace(-xlim, xlim, 1000)
    asy_hist = norm.pdf(pts, scale = sigma)
    line, = ax.plot(pts,asy_hist,'-',color = 'red', label = 'Asymptotic')
    
    ax.set_xlim(-xlim, xlim)
    ax.legend(loc="upper right")
    
    def update(frame):
        counts, _ = np.histogram(results['saved_delta_bars'][frame,:,dim] * np.sqrt(results['save_iter'][frame]), 
                                 bins = bins,
                                 density = True)
        for count, patch in zip(counts, patches):
            patch.set_height(count)
    
        ax.set_title(r"$\sqrt{T}(\bar{\Delta}_T)_{%d},T=%s$" % (dim+1,Tlabels[frame]))
        return patches
    
    ani = FuncAnimation(fig, update, frames=20, interval=100, blit=False)
    ani.save(fname, writer="pillow", fps=10)

    
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

def plot_CI_cover_rates(MRP_params, 
                        results, 
                        confidence = 0.95, 
                        n_samples = 10 ** 3,
                        xmin = 1024, 
                        ymin = 0.8):

    # Individual confidence intervals
    cv = norm.ppf((1+confidence) / 2)
    abs_delta_bars = np.abs(results['saved_delta_bars']) # (n_save, N_trials, d)
    individual_cover = abs_delta_bars < cv * np.sqrt(np.diagonal(results['saved_Lambda_hats'], 
                                                                 axis1 = -1, 
                                                                 axis2 = -2) / results['save_iter'][:,np.newaxis,np.newaxis])
    individual_cover_rates = np.mean(individual_cover, axis = 1)

    # Simultaneous confidence interval
    n_save, N_trials, d = results['saved_delta_bars'].shape
    samples = np.random.normal(size = (d, n_samples))
    Lambda_hats = results['saved_Lambda_hats'] + 1e-8 * np.eye(d)[np.newaxis,np.newaxis,:,:]
    Ls = np.linalg.cholesky(Lambda_hats)
    samples = Ls @ samples #(n_save, N_trials, d, n_samples)
    inf_norms = np.max(np.abs(samples), axis = 2) # (n_save, N_trials, n_samples)
    cvs = np.quantile(inf_norms, q = confidence, axis = 2) # (n_save, N_trials)
    simultaneous_cover = np.max(abs_delta_bars, axis = 2) < (cvs / np.sqrt(results['save_iter'][:,np.newaxis]))
    simultaneous_cover_rates = np.mean(simultaneous_cover, axis = 1)
    

    plt.figure(dpi = 300)
    plt.rcParams['text.usetex'] = True

    for dim in range(MRP_params['d']):
        plt.plot(results['save_iter'],
                 individual_cover_rates[:,dim],
                 label = r'$\mathcal{C}_{' + str(dim+1) + '}$')

    plt.plot(results['save_iter'], simultaneous_cover_rates, label = r'$\mathcal{C}$')

    plt.plot(results['save_iter'], 
             confidence * np.ones_like(results['save_iter']), 
             linestyle = 'dotted', 
             color = 'black',
             label = 'target')
    plt.legend()
    plt.xlim([xmin, results['save_iter'][-1]])
    plt.ylim([ymin,1])
    plt.xscale('log')
    plt.xlabel(r'$T$')
    plt.ylabel('Coverage rate')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))

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
