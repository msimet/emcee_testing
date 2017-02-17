import numpy
import emcee
import make_test_dataset
import scipy.stats
import matplotlib.pyplot as plt

def emcee_init(params, nwalkers):
    """
    Initialize the positions of emcee walkers.
    Inputs:
    - params: a list of tuples where the first item of the tuple is the center of a Gaussian ball
              and the second item is the width of the Gaussian ball
    - nwalkers: how many walkers to initialize the positions for
    """
    ndim = len(params)
    inits = numpy.zeros((nwalkers,ndim))
    for i, (p0, psig) in enumerate(params):
        inits[:, i] = p0 + psig*numpy.random.randn(nwalkers)
    return list(inits)    

def lnprior(theta):
    # Check that the priors are loosely right.
    # CHANGE THIS if you change the defaults in make_test_dataset!
    means = theta[0::2]
    sigmas = theta[1::2]
    if any(means>5) or any(means<-1):
        return -numpy.inf
    elif any(sigmas<-0.5) or any(sigmas>1):
        return -numpy.inf
    else:
        return 0

# boilerplate from http://dan.iel.fm/emcee/current/user/line/

def lnlike(theta, x, y, yerr):
    # Form a bunch of Gaussians from these parameters and compare to the data.
    # This works with only one call to scipy.stats.norm.pdf, after some numpy broadcasting magic.
    means = theta[0::2]
    sigmas = theta[1::2]
    xstack = numpy.tile(x, (len(means),1))
    means = numpy.tile(means, (len(x),1)).T
    sigmas = numpy.tile(sigmas, (len(x),1)).T
    models = scipy.stats.norm.pdf(xstack, loc=means, scale=sigmas)
    return -numpy.sum((models-y)**2/(2*yerr**2))
    
def lnprob(theta, x, y, yerr):
    # Combine likelihood and prior
    lp = lnprior(theta)
    if not numpy.isfinite(lp):
        return -numpy.inf
    like = lnlike(theta, x, y, yerr)
    if numpy.any(numpy.isnan(like)):
        return -numpy.inf
    return lp + like
    
def run_mcmc(nobj=1000):
    nwalkers, nsteps = 100, 100000


    # Get the list of models to fit.  Turn into 2-d numpy arrays for later broadcasting ease.
    x, list_of_gaussians, truths = make_test_dataset.test_dataset(nobj)
    y_arr = numpy.array(list_of_gaussians)
    yerr_arr = numpy.ones_like(y_arr)

    # Set up the emcee sampler
    p0 = emcee_init([(2, 0.1), (0.2, 0.1)]*nobj, nwalkers)
    ndim = len(p0[0])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y_arr, yerr_arr)) 
    sampler.run_mcmc(p0, nsteps)
    numpy.save('chain_%i.npy'%nsteps, sampler.chain)
    
    #Check for correctness.  Look at the truth values & compare.
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    vals = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*numpy.percentile(samples, [16, 50, 84],
                                                axis=0))) 
    vals = numpy.array(vals)
    y_fitted = vals[:,0]
    y_low_err = vals[:,2]
    y_hi_err = vals[:,1]
    truths = truths.flatten()
    plt.plot(truths[0::2], truths[0::2], color='black')
    plt.errorbar(truths[0::2], y_fitted[0::2], yerr=[y_low_err[0::2], y_hi_err[0::2]], linestyle='None', marker='o')
    plt.savefig('mcmc_fit_vs_truth_mass_%i.png'%nsteps)
    plt.clf()
    plt.plot(truths[1::2], truths[1::2], color='black')
    plt.errorbar(truths[1::2], y_fitted[1::2], yerr=[y_low_err[1::2], y_hi_err[1::2]], linestyle='None', marker='o')
    plt.savefig('mcmc_fit_vs_truth_sigma%i.png'%nsteps)
    
    means = y_fitted[0::2]
    sigmas = y_fitted[1::2]
    plt.clf()
    n, bins, patches = plt.hist(means, 20)
    plt.savefig('mcmc_means.png')
    plt.clf()
    plt.hist(sigmas, 20)
    plt.savefig('mcmc_sigmas.png')
    
    # Check for progress
    all_samples = sampler.chain.reshape((-1, ndim))   
    
    print "Average acceptance fraction was %f (should be 0.25-0.5)"%numpy.mean(sampler.acceptance_fraction)
    
    corr_per_param = sampler.get_autocorr_time()
    plt.clf()
    plt.hist(corr_per_param, 30)
    plt.savefig('autocorrelation_per_param.png')
    
if __name__=='__main__':
    import sys
    if len(sys.argv)>1:
        run_mcmc(int(sys.argv[1]))
    else:
        run_mcmc()

