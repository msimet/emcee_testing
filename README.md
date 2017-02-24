The problem
-----------

Originally, this project was intended to stress-test MCMC algorithms for a possible hierarchical fit as in [Lieu et al. 2017](https://github.com/msimet/emcee_testing.git), but with many clusters (order thousands).  The basic characteristics of the problem:
- We want to use an MCMC chain to find distributions for thousands of parameters
- The structure of the data set we're fitting is a set of one-dimensional functions
- The functions can be fit by similar models
- The parameters of the functions are mostly uncorrelated from one function to another (so the covariance matrix between 1D functions is small)--but there are hyperparameters such that the parameters of those functions are drawn from a distribution.

This roughly describes a set of individual cluster weak lensing profiles, where there's a general shape with known parameters, but any two clusters will have widely varying parameters from each other; we just know that, overall, they're drawn from a mass function and a concentration-mass relation of some kind, so the correlation is small but not zero.

The test data set
-----------------

The file `make_test_dataset.py` includes a function `test_dataset()`.  The documentation for that function describes all its inputs.  As an overview, though, it generates a set of Gaussian functions sampled on a grid of x values.  The means and widths of those individual Gaussians are themselves drawn from a 2D Gaussian distribution with a small covariance, giving us hyperparameters.  

From this function, we get a vector of x points (the x values we will give as inputs to the MCMC chain) and a 2d array of y points, where each row of the y array is a Gaussian with its own (mu, sigma) which we will fit using the MCMC chain, and where each column is a sample from that row's Gaussian at the corresponding point from x.  The mu and sigma are like the mass and concentration of the individual clusters.

What's more, if we do things correctly, those mu and sigma values that we fit will themselves be distributed like Gaussians, using the input parameters we gave the function.  These are the hyperparameters, analogous to the parameters of the mass function and the mass-concentration relation we would want to infer with a hierarchical model.

The fitting procedure
---------------------

run_mcmc.py includes some functions to actually run MCMC chains using the `emcee` package [Foreman-Mackey et al, available here](http://dan.iel.fm/emcee/).  At the moment, this does no hierarchical work--it just fits the parameters for each individual cluster.

Results
-------
We tested the scripts using 10 Gaussian functions sampled at 7 x-points each, 100 walkers, and varying numbers of steps.  

- Using 1000 steps, we were not able to get good fits for most of the 20 parameters.  The comparison to the truth values was off, and the internal bookkeeping of the ensemble sampler indicated we hadn't continued long enough to get a reliable estimate of the autocorrelation length--a valuable indicator of whether the chains had converged.  The acceptance fraction was 12% (we would like 25-50%).  Runtime was 11 seconds.

- Using 10000 steps, we got good fits on the means, but not good fits on the sigmas.  We still could not get a reliable estimate of the autocorrelation length.  The acceptance fraction dropped to 10%. Runtime was 70 seconds.

- Using 100000 steps, we still got good fits on the means, and still failed on the sigmas. (Given the coarse x-binning, it is unclear how well we would ever be able to do on the sigmas.)  We still could not get a reliable estimate of the autocorrelation length.  The acceptance fraction was slightly below 10%.  Runtime was 11 minutes.

Given that 100,000 steps failed to produce good results for only 10 objects, using a very fast likelihood estimation method, we decided it was unlikely that this procedure would work for thousands of objects and have not yet investigated further.  However, it is possible that the coarse x-binning was partly to blame, so future work may sample the Gaussians more finely to see if the convergence is better.  The dominant time component is the evaluation of the log likelihood, and this is fully vectorized, so increasing the number of x points should change the runtime by the same ratio.