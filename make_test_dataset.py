import numpy
import matplotlib.pyplot as plt
import scipy.stats

def test_dataset(n_gaussians=1000, n_x_points = 7, mean_of_means=2, sigma_of_means=0.2, 
                 mean_of_sigmas=0.3, sigma_of_sigmas=0.15, make_plots=False):
    """
    Generate a set of Gaussians.  The Gaussians have a mean drawn from a normal distribution with 
    mean `mean_of_means` and width `sigma_of_means` and a width drawn from a normal distribution
    with mean `mean_of_sigmas` and width `sigma_of_sigmas`, ie, with 4 hyperparameters describing
    the distribution of the parameters.
    
    There is a slight covariance between the means and sigmas, with the off-diagonal terms being 10%
    of the power of the diagonals:
    cov = [[sigma_of_mean**2                  0.1*sigma_of_mean*sigma_of_sigma  ]
           [0.1*sigma_of_mean*sigma_of_sigma  sigma_of_sigma**2                 ]]
    
    Inputs
    ------
     - n_gaussians      : the number of Gaussians to return [default 1000]
     - n_x_points       : the number of x points in [1.5, 2.5] at which the Gaussians will be 
                          sampled [default 7]
     - mean_of_means    : the mean of means of the Gaussians, ie a hyperparameter [default 2]
     - sigma_of_means   : the sigma of means of the Gaussians [default 0.2]
     - mean_of_sigmas   : the mean of sigmas of the Gaussians [default 0.3]
     - sigma_of_sigmas  : the sigma of sigmas of the Gaussians [default 0.15]
     - make_plots       : generate diagnostic plots [default False]
     
    Returns
    -------
    x: the x points at which the Gaussians are sampled, shape (n_x_points,)
    list_of_models: a n_gaussians-element list of Gaussian pdfs with the same shape as x, above
    means: a (n_gaussians, 2)-dimensional array with the means in [:,0] and sigmas in [:,1]
    """
    
    # Generate some covariant means and sigmas             
    covariance = numpy.array([[sigma_of_means**2, 0.1*sigma_of_means*sigma_of_sigmas], 
                              [0.1*sigma_of_means*sigma_of_sigmas, sigma_of_sigmas**2]])
    means = numpy.random.multivariate_normal((mean_of_means, mean_of_sigmas), covariance, size=n_gaussians)
    
    
    if make_plots:
        print means.shape
        # Scatter plot of means vs sigmas [look for correlation]
        plt.plot(means[:,0], means[:,1], 'r.')
        plt.savefig('means_vs_sigmas.png')
        
        plt.clf()
        # Histograms to check for individual Gaussian-ness of the means and sigmas
        plt.hist(means[:,0], 20)
        plt.savefig('means.png')
        plt.clf()
        
        plt.hist(means[:,1], 20)
        plt.savefig('sigmas.png')
        plt.clf()

    x = numpy.linspace(1.5, 2.5, num=n_x_points)
    normal_pdf = scipy.stats.norm
    list_of_models = [normal_pdf.pdf(x, loc=mean, scale=sigma) for mean, sigma in means]
    if make_plots:
        # Plot the models themselves
        [plt.plot(x, model, alpha=0.2) for model in list_of_models]
        plt.savefig('functions.png')
    return x, list_of_models, means


if __name__=='__main__':
    test_dataset(make_plots=True)
