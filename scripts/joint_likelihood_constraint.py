# %load /home/ugiri/kineticsz/scripts/constrain_quijote.py
import emcee
import numpy
import nestle
import pathlib
import kineticsz
from rich import print
from multiprocessing import Pool

cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_quijote.json')
alpha = cmb.interpolate_transfer_function()
powerspectrum_function = cmb.interpolate_matter_powerspectrum()

def fnl_prefactor_function(parameters: numpy.ndarray, k: numpy.ndarray, deltac: float=1.42, p: float=1) -> float:
    """ This functtion models the theoretical bias between matter and halos as a function of parameters
        bg, bv(optional) and fnl
    Arguments:
        parameters {numpy.ndarray} -- array containing parameter values
        k {numpy.ndarray} -- array of k values where bias is desired
    Keyword Arguments:
        deltac {float} -- critical threshold for collapse (default: {1.686})
        p {float} --  (default: {1})
    Returns:
        float -- total galaxy bias
    """
    if len(parameters) == 2:
        bias, fnl = parameters
    elif len(parameters) == 3:
        bias, bv, fnl = parameters
    else:
        raise Exception('The number of parameters provided to fnl_prefactor is wrong. The number must be 2 or 3')
    coefficient = (bias + 2*deltac*(bias - p)*fnl/alpha(k))
    return coefficient


def velocity_prefactor_function(parameters: numpy.ndarray, k: numpy.ndarray, kz: numpy.ndarray, redshift:float=2) -> complex:
    """The velocity prefactor which connects linear matter density to velocity modes
    Arguments:
        parameters {numpy.ndarray} -- array containing theoretical parameters which determine the prefactor
    Keyword Arguments:
        redshift {int} -- redshift at which to perform the coefficient calculation (default: {2})
    Returns:
        complex -- The coefficient of type complex
    """
    _, bv, _ = parameters
    f = cmb.model.scale_independent_growth_factor_f(redshift)
    a = 1./(1.+redshift)
    H = cmb.model.Hubble(redshift)*cmb.c
    coefficient = (1j*bv*f*a*H*kz/k**2)
    return coefficient


def density_prefactor_function(parameters: numpy.ndarray, k: numpy.ndarray, kz: numpy.ndarray, redshift:float=2) -> complex:
    """The velocity prefactor which connects linear matter density to velocity modes
    Arguments:
        parameters {numpy.ndarray} -- array containing theoretical parameters which determine the prefactor
    Keyword Arguments:
        redshift {int} -- redshift at which to perform the coefficient calculation (default: {2})
    Returns:
        complex -- The coefficient of type complex
    """
    return parameters[1]


def halo_covariance_likelihood(parameters):
    """2D covariance matrix for MCMC analysis
    Arguments:
        parameters {numpy.ndarray} -- array containing the bias parameters and fnl
    Returns:
         -- negative log likelihood value
   """
    
    fnl_prefactor = fnl_prefactor_function(parameters=parameters, k=k)

    phh = fnl_prefactor**2 * powerspectrum_function(k)
    summation = 0
    covariance = numpy.zeros((1,1), dtype=numpy.complex128); 
    
    for i in range(0, hk_modes*num_files, 1):
        covariance[0,0] = phh[i] + shot_noise
        determinant = numpy.log(numpy.linalg.det(covariance))
        inverse = numpy.linalg.inv(covariance)
        d = numpy.array([(hr[i] + 1j*hi[i])])
        summation += 0.5*(determinant + numpy.matmul(numpy.matmul(d.conjugate(), inverse), d))
    
    return -summation.real + flat_prior(parameters)

def velocity_covariance_likelihood(parameters):
    """2D covariance matrix for MCMC analysis
    Arguments:
        parameters {numpy.ndarray} -- array containing the bias parameters and fnl

    Returns:
         -- negative log likelihood value
   """

    fnl_prefactor = fnl_prefactor_function(parameters=parameters, k=k)
    velocity_prefactor = velocity_prefactor_function(parameters=parameters, k=k, kz=kz)
    powerspectrum = powerspectrum_function(k)
    phh = fnl_prefactor**2 * powerspectrum
    phv = fnl_prefactor * velocity_prefactor * powerspectrum
    pvv = velocity_prefactor * velocity_prefactor.conjugate() * powerspectrum

    nvv = numpy.abs((nr + 1j*ni)*(nr + 1j*ni).conjugate())

    covariance = numpy.zeros((2,2), dtype=numpy.complex128); 
    summation = 0

    for i in range(vk_modes*num_files):
        covariance[0,0] = phh[i] + shot_noise
        covariance[0,1] = phv[i].conjugate()
        covariance[1,0] = phv[i]
        covariance[1,1] = pvv[i] + nvv[i]
        determinant = numpy.log(numpy.linalg.det(covariance))
        inverse = numpy.linalg.inv(covariance)
        d = numpy.array([(hr[i] + 1j*hi[i]), (vr[i] + 1j*vi[i])])
        summation += 0.5*(determinant + numpy.matmul(numpy.matmul(d.conjugate(), inverse), d))
    return -summation.real + flat_prior(parameters)

def matter_covariance_likelihood(parameters):
    """2D covariance matrix for MCMC analysis
    Arguments:
        parameters {numpy.ndarray} -- array containing the bias parameters and fnl

    Returns:
         -- negative log likelihood value
   """

    fnl_prefactor = fnl_prefactor_function(parameters=parameters, k=k)
    #velocity_prefactor = density_prefactor_function(parameters=parameters, k=k, kz=kz)
    velocity_prefactor = 0.99
    powerspectrum = powerspectrum_function(k)
    phh = fnl_prefactor**2 * powerspectrum
    phv = fnl_prefactor * velocity_prefactor * powerspectrum
    pvv =  velocity_prefactor**2*powerspectrum

    covariance = numpy.zeros((2,2), dtype=numpy.complex128); 
    summation = 0

    for i in range(vk_modes*num_files):
        covariance[0,0] = phh[i] + shot_noise
        covariance[0,1] = phv[i].conjugate()
        covariance[1,0] = phv[i]
        covariance[1,1] = pvv[i] + 1 
        determinant = numpy.log(numpy.linalg.det(covariance))
        inverse = numpy.linalg.inv(covariance)
        d = numpy.array([(hr[i] + 1j*hi[i]), (dr[i] + 1j*di[i])])
        summation += 0.5*(determinant + numpy.matmul(numpy.matmul(d.conjugate(), inverse), d))

    return -summation.real + flat_prior(parameters)

mode_files = sorted(list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/modes.npz')))[:]
num_files = len(mode_files)
vk_modes = hk_modes = 58
k, kz, hr, dr, vr, qr, nr, hi, di, vi, qi, ni = (numpy.zeros((len(mode_files), vk_modes), dtype=numpy.float64) for _ in range(12))
valid = 0

for i,name in enumerate(mode_files):
    try: 
        handle = numpy.load(name)
        valid += 1
    except: 
        continue
    k[i,:], kz[i,:] = (handle[x][:hk_modes].real for x in ('k', 'kz'))
    hr[i,:], dr[i,:], vr[i,:], nr[i,:], qr[i,:] = (handle[x][:hk_modes].real for x in ('hm', 'dm', 'vm', 'nm', 'qm'))
    hi[i,:], di[i,:], vi[i,:], ni[i,:], qi[i,:] = (handle[x][:hk_modes].imag for x in ('hm', 'dm', 'vm', 'nm', 'qm'))

print('Number of mode files : {}'.format(num_files))
k, kz, hr, dr, vr, nr, qr, hi, di, vi, ni, qi = (x.flatten() for x in (k, kz, hr, dr, vr, nr, qr, hi, di, vi, ni, qi))
shot_noise = 4000

def random_blob(initial):
    '''This function creates a blob of walkers around a given initial guess'''
    #ndim is number of parameters while nwalkers is number of walkers to be used for emcee
    ndim, nwalkers = initial.size, 50
    #initial_walkers stores the initial poisition of nwalkers in ndim dimensional space
    initial_walkers = numpy.zeros((nwalkers, ndim))
    #Initialize the starting walker position with positions in a blob around the initial guess
    for i in range(nwalkers):
        while numpy.all(initial_walkers[i,:] == 0):
            #sample a walker from a gaussian around the fitburst based initial guess using a covariance of 1e-10
            walker = initial + numpy.random.random(ndim)
            #numpy.random.multivariate_normal(initial, numpy.diag(numpy.full(ndim, 1e-10)))
            #if the simulated ndim vector is a valid data point, accept it as a valid walker
            if (0 < walker[0] < 10) and (0 - 100 < walker[-1] < (0 + 100)):
                initial_walkers[i,:] = walker[:]
    #return the walkers
    return initial_walkers

def flat_prior(parameters):
    if not ((1 < parameters[0] < 5) and ((0 - 500) < parameters[-1] < (0 + 500))):
        return -numpy.inf
    else:
        return 0

def prior_transform(x):
    if len(x) == 3:
        return numpy.array([4*x[0]+1, 2*x[1], 1000*x[2]-500])
    elif len(x) == 2:
        return numpy.array([4*x[0]+1, 1000*x[-1]-500])

def run_nestle(likelihood, position, args):
    result = nestle.sample(loglikelihood=likelihood, prior_transform=prior_transform, ndim=position.size, logl_args=args,       npoints=1000, callback=nestle.print_progress)
    print(result.summary())
    

def run_mcmc(likelihood, position, backend_name=''):
    with Pool() as pool:
        backend = emcee.backends.HDFBackend(filename=backend_name)
        sampler = emcee.EnsembleSampler(50, position.shape[1], likelihood, backend=backend, pool=pool)
        sampler.run_mcmc(position, nsteps=5000, progress=True)
        samples = numpy.array(sampler.get_chain(flat=True));
    return samples

initial = numpy.array([3.3, 1,   0])
position = random_blob(initial)
samplesv2d = run_mcmc(likelihood=velocity_covariance_likelihood, position=position,
                      backend_name='/home/ugiri/kineticsz/mcmc_chains/combined_velocity_quijote_58.h5')

samplesh2d = run_mcmc(likelihood=halo_covariance_likelihood, position=position[:,[0,2]],
                      backend_name='/home/ugiri/kineticsz/mcmc_chains/combined_halo_quijote_58.h5')

samplesv2d = run_mcmc(likelihood=velocity_covariance_likelihood, position=position,
                      backend_name='/home/ugiri/kineticsz/mcmc_chains/combined_velocity_quijote_58_true_noise.h5')

samplesq2d = run_mcmc(likelihood=velocity_covariance_likelihood, position=position,
                      backend_name='/home/ugiri/kineticsz/mcmc_chains/combined_momentum_quijote_58_modes.h5')
'''
samplesd2d = run_mcmc(likelihood=matter_covariance_likelihood, position=position[:,[0,2]],
                      backend_name='/home/ugiri/kineticsz/mcmc_chains/combined_density_quijote_58_modes.h5')
'''
if False:
    run_nestle(likelihood=matter_covariance_likelihood, position=initial, args=[k, kz, hr, hi, dr, di, shot_noise])
