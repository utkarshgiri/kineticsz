import h5py
import fire
import json
import numpy
import logging
import classylss
from pathlib import Path
from kineticsz import read
from scipy import interpolate
from  classylss import binding
from rich.logging import RichHandler

logging.basicConfig(level="NOTSET", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


class LPT:

    def __init__(self, class_config :str, boxsize :float, pixels :int, redshift :float, read_hdf5 :str, write_hdf5 :str,
                    fnl :float, seed :int, units='kpch', scaling='trivial', second_order=True):
        
        logger.info(f'Setting the seed to {seed}')
        numpy.random.seed(seed)

        self.class_config, self.read_hdf5, self.write_hdf5 = (Path(x) for x in (class_config, read_hdf5, write_hdf5))
        assert (self.class_config.suffix == '.json') and self.class_config.exists()
        
        with open(self.class_config, 'r') as f:
            self.class_parameters = json.load(f)
            self.class_parameters['z_pk'] = redshift
            if redshift > 999:
                self.class_parameters['output'] = 'mTk'
                del self.class_parameters['l_max_scalars']; 
                del self.class_parameters['lensing'];
                del self.class_parameters['accurate_lensing']        
                del self.class_parameters['non linear']        
        
        self.fnl = fnl
        self.seed = seed
        self.pixels = pixels
        self.scaling = scaling
        self.boxsize = boxsize
        self.redshift = redshift
        self.units = units        
        self.second_order = second_order 
        precision_params = classylss.load_precision('/home/ugiri/github/class_public/pk_ref.pre')
        #self.class_parameters.update(precision_params)
        logger.info('Class configs are: \n {}'.format(self.class_parameters))
        self.classy = binding.ClassEngine(self.class_parameters)
        self.background = binding.Background(self.classy)
        self.spectrum = binding.Spectra(self.classy) 
        self.primordial = binding.Primordial(self.classy)
        

        if self.units == 'mpch':
            self.factor = self.background.h
        elif self.units == 'kpch':
            self.factor = self.background.h*1000
        else:
            self.factor = 1.
        
        self.mpcboxsize = self.boxsize / self.factor

        
    def __call__(self):
        
        handle = read.Snapshot(self.read_hdf5)
        assert numpy.isclose(self.class_parameters['h'], handle.hubble, atol=0.01)
        assert numpy.isclose(self.class_parameters['z_pk'], handle.redshift, atol=0.01)

        self.write_hdf5.write_bytes(self.read_hdf5.read_bytes())
        
        logger.info('Computing displaced position and velocity')
        position, velocity = self.positions(deltak=self.deltak())
         
        position = numpy.array(position*self.factor, dtype=numpy.float32), 
        velocity = numpy.array(velocity, dtype=numpy.float32)
        
        logger.info('Writing out hdf5 file')
        h5handle = h5py.File(self.write_hdf5, 'r+')
        
        h5handle['PartType1']['Coordinates'][:] = position[:]
        h5handle['PartType1']['Velocities'][:] = velocity[:]
        h5handle.close()
        
        self.class_parameters.update({x:y for (x,y) in self.__dict__.items() if isinstance(y, float)})
        with open(self.write_hdf5.parent.joinpath('initial_configuration.json'), 'w+') as f:
            json.dump(self.class_parameters, f)


    def primordial_powerspectrum(self):
        """ Returns an interpolator object which gives powerspectrum in the units of Mpc^3 """
        k = numpy.logspace(-7, 4, 10000)
        scale_invariant_powerpsectrum = self.class_parameters['A_s'] * (k/self.class_parameters['k_pivot'])**(self.class_parameters['n_s']-1.0)
        interpolator = interpolate.interp1d(k, (2*numpy.pi**2/k**3) * scale_invariant_powerpsectrum, fill_value='extrapolate')
        return interpolator


    def transfer_primordial_potential_to_cdm(self, field='d_tot'):
        """ A function which returns an interpolator for 
        transfer function.
        Args:
            field (str): primordial field of interest.
                Default is 'phi'
            redshift (float): redshift of transfer function.
                Default is the configuration redshift
        Returns:
            An intterpolator that takes k value in 1/Mpc and
            returns the corresponding transfer function
            interpolator for the given field. """
        
        if self.scaling == 'trivial':
            logger.info('Trivial scaling. Using scale factor to scale the transfer function')
            scaling = 1./(1+self.redshift)
        else:
            logger.info('Using growth factor to scale the transfer function')
            scaling = self.background.scale_independent_growth_factor(self.redshift)/self.background.scale_independent_growth_factor(0)
        Tk = self.spectrum.get_transfer(z=0)
        
        Tk = interpolate.interp1d(Tk['k']*self.background.h, scaling*Tk[field], fill_value='extrapolate')
        return Tk


    def deltak(self):
        """ A function which samples phik modes on a 3D k-space grid
        to be used fro creating initial condition for N-body
        simulation

        Args:
            boxsize (float): box size in Mpc
            gridspacing (float): spacing of grids; boxsize/gridsize
            redshift (float): redshift at which the initial field is
                to be generated
            fnl (float): value of fnl. Default is 0

        Returns:
            The fourier space field phik """
        midpoint = int(self.pixels/2)
        k = numpy.zeros(shape=(self.pixels, self.pixels, midpoint+1))
        frequency = 2 * numpy.pi * numpy.fft.fftfreq(n=self.pixels, d=self.mpcboxsize/self.pixels)
        for i in range(self.pixels):
            for j in range(self.pixels):
                k[i,j,:] = numpy.sqrt(frequency[i]**2 + frequency[j]**2 + frequency[:(midpoint+1)]**2)
        
        powerspectrum = self.primordial_powerspectrum()(k)
        Tk = self.transfer_primordial_potential_to_cdm()(k) #in units of k
         
        sdev = numpy.sqrt(powerspectrum*(self.pixels**6/self.mpcboxsize**3)/2.0)
        real = numpy.random.normal(loc=0, scale=sdev, size=sdev.shape)
        imag = numpy.random.normal(loc=0, scale=sdev, size=sdev.shape)
        phik = real + 1j*imag
       
        phik[0,0,0] = 0; 
        phik[midpoint,:,:] = 0; phik[:,midpoint,:] = 0; phik[:,:,midpoint] = 0

        #Adding non-gaussianity
        phik = self.hermitianize(phik)
        phi = numpy.fft.irfftn(phik)
        phi = phi + self.fnl*(phi**2 - numpy.mean(phi*phi))
        phik = numpy.fft.rfftn(phi)
        phik = self.hermitianize(phik)
       
        phik = Tk*phik
        phik = self.hermitianize(phik)
        
        return phik


    def displacement(self, deltak):
        """ Function to compute the displacement field from potential field
        Args:
            phik (ndarray): A 3D array containing the potential field in fourier space
            boxsize (float): Size of the box
            gridspacing (float): spacing of grids
            fnl (float): Value of fnl parameter. Defaults to 0
        Returns:
            displacement field vector for x, y and z.  """
        
        midpoint = int(self.pixels/2)
        frequency = 2 * numpy.pi * numpy.fft.fftfreq(n=self.pixels, d=self.mpcboxsize/self.pixels)
        kx, ky, kz = numpy.meshgrid(frequency, frequency, frequency[:midpoint+1])
        k = numpy.sqrt(kx**2 + ky**2 + kz**2)
      
        phik = numpy.divide(-deltak, k**2, out=numpy.zeros_like(deltak), where=k!=0)
        phikx = -1j*kx*phik; phiky = -1j*ky*phik; phikz = -1j*kz*phik
         
        psix = numpy.fft.irfftn(self.hermitianize(phikx))
        psiy = numpy.fft.irfftn(self.hermitianize(phiky))
        psiz = numpy.fft.irfftn(self.hermitianize(phikz))
        
        return (psix, psiy, psiz)


    def second_order_displacement(self, deltak):
        """ Function to compute the displacement field from potential field
        Args:
            phik (ndarray): A 3D array containing the potential field in fourier space
            boxsize (float): Size of the box
            gridspacing (float): spacing of grids
            fnl (float): Value of fnl parameter. Defaults to 0
        Returns:
            displacement field vector for x, y and z.  """
        
        omegam = self.background.Omega_m(self.redshift)
        midpoint = int(self.pixels/2)
        frequency = 2 * numpy.pi * numpy.fft.fftfreq(n=self.pixels, d=self.mpcboxsize/self.pixels)
        kx, ky, kz = numpy.meshgrid(frequency, frequency, frequency[:midpoint+1])
        k = numpy.sqrt(kx**2 + ky**2 + kz**2)
      
        phik = numpy.divide(-deltak, k**2, out=numpy.zeros_like(deltak), where=k!=0)
        phikxkx = numpy.fft.irfftn(kx*kx*phik); 
        phikxky = numpy.fft.irfftn(kx*ky*phik);
        phikxkz = numpy.fft.irfftn(kx*kz*phik);
        phikyky = numpy.fft.irfftn(ky*ky*phik);
        phikykz = numpy.fft.irfftn(ky*kz*phik);
        phikzkz = numpy.fft.irfftn(kz*kz*phik);

        Fq = phikyky*phikxkx + phikzkz*phikyky + phikzkz*phikxkx - phikykz**2 - phikxkz**2 - phikxky**2
        del phikxkx; del phikxky; del phikxkz; del phikyky; del phikzkz; del phikykz
        phi2k = (3./7)*omegam**(-1/143)*numpy.divide(numpy.fft.rfftn(Fq), k**2, out=numpy.zeros_like(deltak), where=k!=0)
        phi2k = self.hermitianize(phi2k)
        
        psix = numpy.fft.irfftn(self.hermitianize(1j*kx*phi2k))
        psiy = numpy.fft.irfftn(self.hermitianize(1j*ky*phi2k))
        psiz = numpy.fft.irfftn(self.hermitianize(1j*kz*phi2k))
        
        return (psix, psiy, psiz)


    def velocities(self, psix, psiy, psiz):
        """ Function to calculate velocity field from displacement vector fields
        Args:
            psix (ndarray): displacement field along x
            psiy (ndarray): displacement field along y
            psiz (ndarray): displacement field along z
        Returns:
            tuple containing velocities (vx, vy, vz) for all the particles """

        #f = self.cosmology.scale_independent_growth_factor_f(self.redshift)
        f = self.background.scale_independent_growth_rate(self.redshift)
        h = self.background.hubble_function(self.redshift)*3e5
        #h = self.classy.Hubble(self.redshift)*3e5
        a = 1./(1. + self.redshift)
        factor = f * a * h /numpy.sqrt(a)
        
        return (psix.flatten()*factor, psiy.flatten()*factor, psiz.flatten()*factor)

    def second_order_velocities(self, psi2x, psi2y, psi2z):
        """ Function to calculate velocity field from displacement vector fields
        Args:
            psix (ndarray): displacement field along x
            psiy (ndarray): displacement field along y
            psiz (ndarray): displacement field along z
        Returns:
            tuple containing velocities (vx, vy, vz) for all the particles """

        #f = self.cosmology.scale_independent_growth_factor_f(self.redshift)
        f = self.background.scale_independent_growth_rate(self.redshift)
        h = self.background.hubble_function(self.redshift)*3e5
        #h = self.classy.Hubble(self.redshift)*3e5
        a = 1./(1. + self.redshift)
        factor = f * a * h /numpy.sqrt(a)
        f2 = 2*(self.background.Omega_m(self.redshift))**(6./11) * a * h / numpy.sqrt(a)
        return (psi2x.flatten()*f2, psi2y.flatten()*f2, psi2z.flatten()*f2)


    def positions(self, deltak, second_order=True):
        """ Function which takes potential field at a redshift and returns initial condition position and velocity
        Args:
            phik (ndarray): Potential field at initial
                redshift.
            boxsize (float): Size of the box
            gridspacing (float): Spacing of the grid
        Returns:
            (position, velocity) for the particles """
        
        psix, psiy, psiz = self.displacement(deltak)
        if self.second_order:
            psi2x, psi2y, psi2z = self.second_order_displacement(deltak)
            psix = psix + psi2x
            psiy = psiy + psi2y
            psiz = psiz + psi2z

        gridspacing = self.mpcboxsize/self.pixels 
        space = numpy.arange(start=0.0+gridspacing/2, stop=(self.mpcboxsize+gridspacing/2), step=gridspacing)
        x, y, z = numpy.meshgrid(space, space, space)
        x += psix; y += psiy; z += psiz

        position = numpy.column_stack([x.flatten(), y.flatten(), z.flatten()])
        del x; del y; del z;
        velocity = numpy.column_stack(self.velocities(psix, psiy, psiz))
        del psix; del psiy; del psiz
        if self.second_order:
            velocity = velocity + numpy.column_stack(self.second_order_velocities(psi2x, psi2y, psi2z))
        
        position[position<0] += self.mpcboxsize 
        position[position>self.mpcboxsize] -= self.mpcboxsize
        
        return position, velocity

    def hermitianize(self, x):
        """A function that self.hermitianizes the fourier array. A cleaner version of hermitianate.
        The logic is taken from:
        `https://github.com/nualamccullagh/zeldovich-bao/` """
        
        pixels = x.shape[0]; midpoint = int(pixels/2)
        for index in [0, midpoint]:
            x[midpoint+1:,1:,index]= numpy.conj(numpy.fliplr(numpy.flipud(x[1:midpoint,1:,index])))
            x[midpoint+1:,0,index] = numpy.conj(x[midpoint-1:0:-1,0,index])
            x[0,midpoint+1:,index] = numpy.conj(x[0,midpoint-1:0:-1,index])
            x[midpoint,midpoint+1:,index] = numpy.conj(x[midpoint,midpoint-1:0:-1,index])
        return x

if __name__ == "__main__":
    fire.Fire(LPT)

