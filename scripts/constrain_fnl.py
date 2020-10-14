import fire
import emcee
import scipy
import numpy
import logging
import utensils
from typing import Union
from pathlib import Path, PosixPath
from matplotlib import pyplot as plt
from rich.logging import RichHandler
from nbodykit.lab import ArrayCatalog, ArrayMesh
from kineticsz import utils, reconstruct, pycmb, Simulation, Reconstructor

plt.rcParams['text.usetex'] = True
logging.getLogger("CatalogMesh").setLevel(logging.WARNING)
logging.getLogger("MeshSource").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

class MCMC:

    def __init__(self, simulation :Union[Simulation, str], noise_simulation :Union[Simulation, str]=None, true_fnl :int=0, hk_modes :int=250,
                 vk_modes:int=120, voxels:int=1024, num_burn :int=500, num_step :int=2000, nwalkers :int=50, use_matter :bool=False, clear_cache=False,
                 hk_kmax :float=0.012, load=True, vk_kmax :float=0.012, dtype :str='float', white_noise=1.0, beam_width=1.0, in_units='kpch',
                 out_units='mpc', mu_min=0.0, add_noise=True, class_params='/home/ugiri/kineticsz/ics/configuration/class_quijote.json', ksmax=1e2,
                 minmass=1e0, maxmass=1e20, interlacing=False, resampler='CIC', modes_file='flattened_modes.npz', distort_filter=True, perform_mcmc=False):
        
        self.hk_kmax  = hk_kmax
        self.vk_kmax  = vk_kmax
        self.true_fnl  = true_fnl
        self.hk_modes  = hk_modes
        self.vk_modes  = vk_modes
        self.num_burn  = num_burn
        self.num_step  = num_step
        self.nwalkers  = nwalkers
         
        self.cmb = pycmb.CMB(class_params, units=out_units)
        self.alpha = self.cmb.interpolate_transfer_function()
        self.powerspectrum = self.cmb.interpolate_matter_powerspectrum()
        
        if isinstance(simulation, Simulation): 
            self.simulation = simulation    
        else:
            self.simulation = Simulation(dirname=simulation, voxels=voxels, in_units=in_units, out_units=out_units, 
                                            class_parameters=self.cmb.parameters, minmass=minmass, maxmass=maxmass)
        
        if noise_simulation is not None:
            self.noise_simulation = Simulation(dirname=noise_simulation, voxels=voxels, in_units=in_units, out_units=out_units, class_parameters=self.cmb.parameters)
        else:
            self.noise_simulation = noise_simulation
        
        self.load = load
        self.voxels  = voxels
        self.resampler = resampler
        self.interlacing = interlacing
        self.redshift = self.simulation.snapshot.redshift
        self.boxsize = float(self.simulation.snapshot.boxsize)
        self.number_density = self.simulation.number_density
        self.dtype = numpy.float32 if dtype == 'float' else numpy.float64
        self.clear_cache = clear_cache
        self.modes_file = modes_file
        self.perform_mcmc = perform_mcmc 
        self.distort_filter = distort_filter

        if out_units == 'mpch':
            self.factor = 1/self.cmb.parameters['h']
        else:
            self.factor = 1
        self.reconstructor = Reconstructor(self.simulation, self.noise_simulation, ksmax=ksmax, class_params=class_params, 
                load=self.load, interlacing=self.interlacing, resampler=self.resampler, distort_filter=self.distort_filter)

        logger.info('Settings used: \n {}'.format({x:y for (x,y) in self.__dict__.items() if isinstance(y, (float, int, bool, str))}))



    def fnl_prefactor(self, parameters: numpy.ndarray, k: numpy.ndarray, deltac: float=1.42, p: float=1) -> float:
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
        coefficient = (bias + 2*1.42*(bias - 1.0)*fnl/self.alpha(k))
        return coefficient


    def velocity_prefactor(self, parameters: numpy.ndarray, k: numpy.ndarray, kz: numpy.ndarray, redshift:float=2) -> complex:
        """The velocity prefactor which connects linear matter density to velocity modes
        Arguments:
            parameters {numpy.ndarray} -- array containing theoretical parameters which determine the prefactor
        Keyword Arguments:
            redshift {int} -- redshift at which to perform the coefficient calculation (default: {2})
        Returns:
            complex -- The coefficient of type complex
        """
        bias, bv, fnl = parameters
        f = self.cmb.model.scale_independent_growth_factor_f(redshift)
        a = 1./(1.+redshift)
        H = self.cmb.model.Hubble(redshift)*self.cmb.c*self.factor
        coefficient = (1j*bv*f*a*H*kz/k**2)
        return coefficient


    def density_prefactor(self, parameters: numpy.ndarray, k: numpy.ndarray, kz: numpy.ndarray, redshift:float=2) -> complex:
        """The velocity prefactor which connects linear matter density to velocity modes
        Arguments:
            parameters {numpy.ndarray} -- array containing theoretical parameters which determine the prefactor
        Keyword Arguments:
            redshift {int} -- redshift at which to perform the coefficient calculation (default: {2})
        Returns:
            complex -- The coefficient of type complex
        """
        bias, bv, fnl = parameters
        f = self.cmb.model.scale_independent_growth_factor_f(redshift)
        a = 1./(1.+redshift)
        H = self.cmb.model.Hubble(redshift)*self.cmb.c
        coefficient = bv
        return coefficient


    def halo_covariance_likelihood(self, parameters, k, kz, hr, hi, vr, vi, shot_noise):
        """2D covariance matrix for MCMC analysis
        Arguments:
            parameters {numpy.ndarray} -- array containing the bias parameters and fnl
            k {numpy.ndarray} -- absolute value of k modes used in the calculation
            kz {numpy.ndarray} -- The radial component of k vector
            hr {numpy.ndarray} -- real component of halo density mode
            hi {numpy.ndarray} -- imaginary component of halo density mode
            shot_noise {float} -- halo shot noise
        Returns:
             -- negative log likelihood value
       """

        if not ((-10 < parameters[0] < 10) and ((self.true_fnl -500) < parameters[-1] < (self.true_fnl + 500))):
            return -numpy.inf

        fnl_prefactor = self.fnl_prefactor(parameters=parameters, k=k)

        phh = fnl_prefactor**2 * self.powerspectrum(k)
        summation = 0.0
        covariance = numpy.zeros((1,1), dtype=numpy.complex128); 
        for i in range(0, self.hk_modes, 1):
            covariance[0,0] = phh[i] + shot_noise
            determinant = numpy.log(numpy.linalg.det(covariance))
            inverse = numpy.linalg.inv(covariance)
            d = numpy.array([(hr[i] + 1j*hi[i])])
            summation += 0.5*(determinant + numpy.matmul(numpy.matmul(d.conjugate(), inverse), d))
        return -summation.real

    def velocity_covariance_likelihood(self, parameters, k, kz, hr, hi, vr, vi, nr, ni, shot_noise):
        """2D covariance matrix for MCMC analysis
        Arguments:
            parameters {numpy.ndarray} -- array containing the bias parameters and fnl
            k {numpy.ndarray} -- absolute value of k modes used in the calculation
            kz {numpy.ndarray} -- The radial component of k vector
            hr {numpy.ndarray} -- real component of halo density mode
            hi {numpy.ndarray} -- imaginary component of halo density mode
            shot_noise {float} -- halo shot noise
        Returns:
             -- negative log likelihood value
       """

        if not ((-10 < parameters[0] < 10) and (0 < parameters[1]) and ((self.true_fnl - 500) < parameters[-1] < (self.true_fnl + 500))):
            return -numpy.inf

        fnl_prefactor = self.fnl_prefactor(parameters=parameters, k=k)
        velocity_prefactor = self.velocity_prefactor(parameters=parameters, k=k, kz=kz)
        powerspectrum = self.powerspectrum(k)
        phh = fnl_prefactor**2 * powerspectrum
        phv = fnl_prefactor * velocity_prefactor * powerspectrum
        pvv = velocity_prefactor * velocity_prefactor.conjugate() * powerspectrum
        
        nvv = numpy.abs((nr + 1j*ni)*(nr + 1j*ni).conjugate())

        covariance = numpy.zeros((2,2), dtype=numpy.complex128); 
        summation = 0

        for i in range(self.vk_modes):
            covariance[0,0] = phh[i] + shot_noise
            covariance[0,1] = phv[i].conjugate()
            covariance[1,0] = phv[i]
            covariance[1,1] = pvv[i] + nvv[i]
            determinant = numpy.log(numpy.linalg.det(covariance))
            inverse = numpy.linalg.inv(covariance)
            d = numpy.array([(hr[i] + 1j*hi[i]), (vr[i] + 1j*vi[i])])
            summation += 0.5*(determinant + numpy.matmul(numpy.matmul(d.conjugate(), inverse), d))
        covariance = numpy.zeros((1,1), dtype=numpy.complex128); 
        for i in range(self.vk_modes, self.hk_modes):
            covariance[0,0] = phh[i] + shot_noise
            determinant = numpy.log(numpy.linalg.det(covariance))
            inverse = numpy.linalg.inv(covariance)
            d = numpy.array([(hr[i] + 1j*hi[i])])
            summation += 0.5*(determinant + numpy.matmul(numpy.matmul(d.conjugate(), inverse), d))

        return -summation.real

    def matter_covariance_likelihood(self, parameters, k, kz, hr, hi, vr, vi, shot_noise):
        """2D covariance matrix for MCMC analysis
        Arguments:
            parameters {numpy.ndarray} -- array containing the bias parameters and fnl
            k {numpy.ndarray} -- absolute value of k modes used in the calculation
            kz {numpy.ndarray} -- The radial component of k vector
            hr {numpy.ndarray} -- real component of halo density mode
            hi {numpy.ndarray} -- imaginary component of halo density mode
            shot_noise {float} -- halo shot noise
        Returns:
             -- negative log likelihood value
       """

        if not ((-10 < parameters[0] < 10) and (0 < parameters[1]) and ((self.true_fnl - 500) < parameters[-1] < (self.true_fnl + 500))):
            return -numpy.inf

        fnl_prefactor = self.fnl_prefactor(parameters=parameters, k=k)
        velocity_prefactor = self.density_prefactor(parameters=parameters, k=k, kz=kz)
        powerspectrum = self.powerspectrum(k)
        phh = fnl_prefactor**2 * powerspectrum
        phv = fnl_prefactor * velocity_prefactor * powerspectrum
        pvv = velocity_prefactor * velocity_prefactor.conjugate() * powerspectrum

        covariance = numpy.zeros((2,2), dtype=numpy.complex128); 
        summation = 0

        for i in range(self.vk_modes):
            covariance[0,0] = phh[i] + shot_noise
            covariance[0,1] = phv[i].conjugate()
            covariance[1,0] = phv[i]
            covariance[1,1] = pvv[i] + 1 
            determinant = numpy.log(numpy.linalg.det(covariance))
            inverse = numpy.linalg.inv(covariance)
            d = numpy.array([(hr[i] + 1j*hi[i]), (vr[i] + 1j*vi[i])])
            summation += 0.5*(determinant + numpy.matmul(numpy.matmul(d.conjugate(), inverse), d))
        covariance = numpy.zeros((1,1), dtype=numpy.complex128); 
        for i in range(self.vk_modes, self.hk_modes):
            covariance[0,0] = phh[i] + shot_noise
            determinant = numpy.log(numpy.linalg.det(covariance))
            inverse = numpy.linalg.inv(covariance)
            d = numpy.array([(hr[i] + 1j*hi[i])])
            summation += 0.5*(determinant + numpy.matmul(numpy.matmul(d.conjugate(), inverse), d))

        return -summation.real


    def unpack_modes(self):
        """This is a helper function to unpack modes for further usage
        Returns:
            [type] -- a tuple of quantities of interest
        """
        shot_noise = self.simulation.rockstar.shot_noise()
        basename = Path(self.simulation.snapshot.dirname).parent

        modes_datafile = self.simulation.dirname.joinpath(self.modes_file)
        if modes_datafile.exists():
            logger.debug('modes datafile exists')
            handle = numpy.load(modes_datafile)
            k, kx, ky, kz, hm, dm, qm, vm, nm = (handle[x] for x in ('k', 'kx', 'ky', 'kz', 'hm', 'dm', 'qm', 'vm', 'nm'))
            return (kx, ky, kz, k, dm, hm, qm, vm, nm, shot_noise)

        nm, vm = self.reconstructor.reconstruct(load=True)
        kx, ky, kz, k, vm, indices = utils.modes_and_indices(vm, boxsize=self.boxsize)
        nm = nm.flatten()[indices]
        dm = self.simulation.snapshot.density_field(load=True).complex.flatten()[indices]
        hm = self.simulation.rockstar.density_field(load=True).complex.flatten()[indices]
        qm = self.simulation.snapshot.radial_momentum_field(load=True).complex.flatten()[indices]
        hm, dm, vm, qm, nm = (x/self.boxsize**1.5 for x in (hm, dm, vm, qm, nm))
 
        numpy.savez(modes_datafile, k=k, kx=kx, ky=ky, kz=kz, hm=hm, dm=dm, qm=qm, vm=vm, nm=nm)
        return (kx, ky, kz, k, dm, hm, qm, vm, nm, shot_noise)


    def plot_diagnostic(self, k, vmodes, qmodes, nmodes):

        plt.loglog(k, (vmodes*vmodes.conjugate()).real, label='velocity')
        plt.loglog(k, (qmodes*qmodes.conjugate()).real, label='momentum')
        plt.loglog(k, (nmodes*nmodes.conjugate()).real, label='noise')
        plt.legend()
        utensils.save_and_upload_plot(filename='diagnostic.pdf')

    def random_blob(self, initial):
        '''This function creates a blob of walkers around a given initial guess'''
        #ndim is number of parameters while nwalkers is number of walkers to be used for emcee
        ndim, nwalkers = initial.size, self.nwalkers
        #initial_walkers stores the initial poisition of nwalkers in ndim dimensional space
        initial_walkers = numpy.zeros((nwalkers, ndim))
        #Initialize the starting walker position with positions in a blob around the initial guess
        for i in range(nwalkers):
            while numpy.all(initial_walkers[i,:] == 0):
                #sample a walker from a gaussian around the fitburst based initial guess using a covariance of 1e-10
                walker = initial + numpy.random.random(ndim)
                #numpy.random.multivariate_normal(initial, numpy.diag(numpy.full(ndim, 1e-10)))
                #if the simulated ndim vector is a valid data point, accept it as a valid walker
                if (0 < walker[0] < 10) and ((self.true_fnl) - 100 < walker[-1] < (self.true_fnl + 100)):
                    initial_walkers[i,:] = walker[:]
        #return the walkers
        return initial_walkers

    def run_mcmc(self, likelihood, position, args, backend_name=''):
        
        backend = emcee.backends.HDFBackend(filename=backend_name)
        #sampler = emcee.EnsembleSampler(self.nwalkers, position.shape[1], likelihood, args=args)
        #position, _, _ = sampler.run_mcmc(position, nsteps=self.num_burn, progress=True)
        sampler = emcee.EnsembleSampler(self.nwalkers, position.shape[1], likelihood, args=args, backend=backend)
        sampler.run_mcmc(position, nsteps=self.num_step, progress=True)
        samples = numpy.array(sampler.get_chain(flat=True));
        return samples


    def __call__(self):
        
        kx, ky, kz, k, dmodes, hmodes, qmodes, vmodes, nmodes, shotnoise = self.unpack_modes()
        if not self.perform_mcmc:
            logging.info('Returning...')
            return
        if self.vk_kmax is not None:
            self.vk_modes = len(k[k<self.vk_kmax])
        if self.hk_kmax is not None:
            self.hk_modes = len(k[k<self.hk_kmax])
        assert self.hk_modes >= self.vk_modes
        kx, ky, kz, k, hmodes = [x[:self.hk_modes] for x in [kx, ky, kz, k, hmodes]]
        dmodes, qmodes, vmodes, nmodes = [x[:self.vk_modes] for x in [dmodes, qmodes, vmodes, nmodes]]
        
        logger.info('Number  of velocity modes {}'.format(self.vk_modes))
        #* Performing 2D MCMC analysis
        #initial contains initial guess for [bias, bv, fnl]
        logger.info('Starting MCMC for reconstructed velocity')
        initial = numpy.array([2, 1, self.true_fnl])
        #position is a random blob of walkers around the initial position
        position = self.random_blob(initial)
        dirnum = Path(str(self.simulation.dirname)).as_posix().split('/')[-1]
        #dirnum = str(self.simulation.dirname).split('/snapdir_001')[0].split('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/')[-1]

        samplesv2d = self.run_mcmc(likelihood=self.velocity_covariance_likelihood, position=position,
                                  args=(k, kz, hmodes.real, hmodes.imag, vmodes.real, vmodes.imag, nmodes.real, nmodes.imag, shotnoise),
                                  backend_name='/home/ugiri/kineticsz/mcmc_chains/velocity_samples_analytic_noise_{}.h5'.format(dirnum))

        logger.info('Starting MCMC for momentum density')
        samplesq2d = self.run_mcmc(likelihood=self.velocity_covariance_likelihood, position=position,
                                  args=(k, kz, hmodes.real, hmodes.imag, qmodes.real, qmodes.imag, nmodes.real/1e3, nmodes.imag/1e3, shotnoise),
                                  backend_name='/home/ugiri/kineticsz/mcmc_chains/{}_{}momentum_samples.h5'.format(dirnum, int(self.boxsize)))
 


        
        logger.info('Starting MCMC for density')
        samplesd2d = self.run_mcmc(likelihood=self.matter_covariance_likelihood, position=position,
                                  args=(k, kz, hmodes.real, hmodes.imag, dmodes.real, dmodes.imag, shotnoise),
                                  backend_name='/home/ugiri/kineticsz/mcmc_chains/density_samples_{}.h5'.format(dirnum))

        print(numpy.median(samplesd2d.reshape(-1,3), axis=0))
        #* Performing 1D MCMC analysis
        logger.info('Starting MCMC for halo modes')
        samples1d = self.run_mcmc(likelihood=self.halo_covariance_likelihood, position=position[:,[0,1]],
                                  args=(k, kz, hmodes.real, hmodes.imag, qmodes.real, qmodes.imag, shotnoise),
                                  backend_name='/home/ugiri/kineticsz/mcmc_chains/halo_samples_{}.h5'.format(dirnum))
        
        #clear cache
        if self.clear_cache:
            for filename in self.simulation.dirname.glob('*.density'):
                if 'momentum' not in str(filename):
                    filename.unlink()



if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    fire.Fire(MCMC)
