import fire
import scipy
import numpy
import logging
import kineticsz
from nbodykit.lab import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Reconstructor:

    def __init__(self, simulation, secondary_simulation=None, ksmax=1e2, load=True, class_params=None,
            interlacing=False, resampler='CIC', distort_filter=False):

        self.load = load
        self.ksmax = ksmax
        self.resampler = resampler
        self.simulation = simulation
        self.interlacing = interlacing
        self.class_params = class_params
        self.distort_filter = distort_filter
        self.secondary_simulation = secondary_simulation

    def Noisemap(self, snapshot):
        """A function to produce 2D temperature map with primary CMB and noise power

        Args:
            snapshot (kineticsz.Snapshot): snapshot for which the map should be produced

        Returns:
            numpy.ndarray: Array containg T map in real space
        """
        pixels = snapshot.voxels 
        cmb = kineticsz.pycmb.CMB(parameters=self.class_params, units=self.simulation.snapshot.out_units)
        chi_star = cmb.Chistar(snapshot.redshift)
        k2d = kineticsz.utils.k2D(snapshot.boxsize, pixels)
        noiseck = numpy.load('/home/ugiri/kineticsz/data/Ck_sw_0.5_beam_1_in_mpc_out_mpc_hr.npz')
        el = noiseck['k']*chi_star
        Cl = (noiseck['k']**2/el**2)*(noiseck['noiseck'] + noiseck['cmbck'])
        Clcmb = Cl*(chi_star*snapshot.boxsize)**2
        std2D = numpy.sqrt(numpy.interp(k2d.flatten(), el/chi_star, Clcmb/2)).reshape(k2d.shape)
        map2D = numpy.random.normal(scale=std2D) + 1j*numpy.random.normal(scale=std2D)
        map2D[int(pixels/2 +1):, 0] = numpy.conj(map2D[1:int(pixels/2),0][::-1])
        
        return (snapshot.voxels/snapshot.boxsize)**2*numpy.fft.irfft2(map2D)

    def reconstruct(self, load=True):
        """Function to reconstruct the radial velocity using kSZ velocity reconstruction

        Args:
            load (bool, optional): [description]. Defaults to True.

        Returns:
            (numpy.ndarray, numpy.ndarray): noise and velocity arrays
        """
        logger.info('Starting Reconstruction ...')
        assert self.simulation.voxels == self.simulation.snapshot.voxels
        vfilename = self.simulation.dirname.joinpath('reconstructed_velocity.density')
        rfilename = self.simulation.dirname.joinpath('residual_noise.density')
        
        pixel_density_factor = (self.simulation.voxels/self.simulation.boxsize)
        nbodykit_factor = (self.simulation.boxsize**3/self.simulation.voxels**3)

        forward = self.simulation.boxsize/self.simulation.voxels 
        backward = 1./forward
        
        momentum_density = self.simulation.snapshot.radial_momentum_field(load=self.load).real
        halo_density = self.simulation.rockstar.density_field(load=self.load).real -1

        phh = self.simulation.rockstar.halo_powerspectrum(load=self.load, interlacing=self.interlacing, resampler=self.resampler)
        pmh = self.simulation.snapshot.matter_halo_powerspectrum(load=self.load, interlacing=self.interlacing, resampler=self.resampler)
        pmm = self.simulation.snapshot.matter_powerspectrum(load=self.load, interlacing=self.interlacing, resampler=self.resampler)
        pqq = self.simulation.snapshot.momentum_powerspectrum(load=self.load, interlacing=self.interlacing, resampler=self.resampler)
        pksz = self.simulation.snapshot.ksz_powerspectrum(load=self.load)
        
        if self.distort_filter: 
            logger.info('Distorting ..')
            pmh.powerspectrum = pmh.powerspectrum*numpy.exp(-pmh.k**2)

        k2d = kineticsz.utils.k2D(self.simulation.boxsize, self.simulation.voxels).astype(numpy.float32)
        k3d = kineticsz.utils.k3D(self.simulation.boxsize, self.simulation.voxels).astype(numpy.float32)
        
        Phh = numpy.interp(k3d.flatten(), phh.k.real[1:-1], phh.powerspectrum.real[1:-1]).reshape(k3d.shape).astype(numpy.float32)
        Pmh = numpy.interp(k3d.flatten(), pmh.k.real[1:-1], pmh.powerspectrum.real[1:-1]).reshape(k3d.shape).astype(numpy.float32)

        cmb = kineticsz.pycmb.CMB(parameters=self.class_params, units=self.simulation.snapshot.out_units)
        kstar = cmb.Kksz(self.simulation.redshift)

        noisepk = numpy.load('/home/ugiri/kineticsz/data/Ck_sw_0.5_beam_1_in_mpc_out_mpc_hr.npz')
        Ckksz = numpy.interp(k2d.flatten(), pksz.k.real[1:-1], (kstar**2*pksz.powerspectrum[1:-1].real)).reshape(k2d.shape)
        Cknoise = numpy.interp(k2d.flatten(), noisepk['k'], (noisepk['cmbck'] + noisepk['noiseck'])).reshape(k2d.shape)
        Cktot = Ckksz + Cknoise

        weight = numpy.where(numpy.abs(k2d>self.ksmax), 0, 1)
        
        noise = 1./(kstar**2*(forward**3*scipy.fft.rfftn(backward**3*scipy.fft.irfftn((Pmh**2/Phh), workers=40) * 
                    backward**2*scipy.fft.irfft2(weight/Cktot)[:,:,None], workers=40)))
        noise = numpy.array(noise, dtype=numpy.complex64)
        noisepk = FFTPower(ArrayMesh(scipy.fft.irfftn(numpy.sqrt(noise*self.simulation.boxsize**3)/nbodykit_factor),
                    BoxSize=self.simulation.boxsize), mode='1d')
        
        noisepk = kineticsz.utils.Powerspectrum(noisepk, simulation=self.simulation, filename='noise')
        cmbmap = self.Noisemap(self.simulation.snapshot)
        Tksz = momentum_density.real.sum(2)
        prefactor = float(kstar*pixel_density_factor**2)
        Tksz = prefactor * Tksz + cmbmap 
        logger.info('Computing signal ...')        
        T = backward**2*scipy.fft.irfft2(forward**2*weight*scipy.fft.rfft2(Tksz)/Cktot)
        D = backward**3*scipy.fft.irfftn(forward**3*scipy.fft.rfftn(halo_density.real)*(Pmh/Phh))
        signal = T[:,:,None]*D
        v = kineticsz.Density(numpy.float32(backward**3*scipy.fft.irfftn(kstar * noise * forward**3*scipy.fft.rfftn(signal))),
                                simulation=self.simulation)
        v.powerspectrum('reconstructed-velocity')
        v.save(vfilename)
        r = kineticsz.Density((v - momentum_density), simulation=self.simulation)
        r.powerspectrum('residual-noise')
        r.save(rfilename)
        logger.info('Reconstruction over')
        return (numpy.sqrt(noise*self.simulation.boxsize**3), scipy.fft.rfftn(v.real))


def reconstruct(dirname, voxels=1024, in_units='kpch', out_units='mpc', class_params=None):
    simulation = kineticsz.Simulation(dirname, in_units=in_units, out_units=out_units)
    r = Reconstructor(simulation, class_params=class_params)    
    r.reconstruct()

if __name__ == '__main__':
    fire.Fire(reconstruct)

