import fire
import numpy
import scipy
import joblib
import pathlib
import logging
import utensils
import kineticsz
from nbodykit.lab import *
from scipy import interpolate
from rich.logging import RichHandler
from kineticsz.utils import Powerspectrum

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def main(noise='N1', voxels=256, simulations=1):
    simulation = kineticsz.Simulation(dirname='/gpfs/ugiri/gadget/size1024/fnl50z127n1/', in_units='kpch', lazy=True, voxels=256)

    cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_quijote.json', units='mpc')
    pmm = cmb.interpolate_linear_powerspectrum(redshift=2)
    kstar = cmb.Kksz(2)
    pixel_density_factor = (simulation.voxels/simulation.boxsize)
    forward = simulation.boxsize/simulation.voxels 
    backward = 1/forward

    def Noisemap():

        pixels = simulation.voxels 
        cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_quijote.json', units='mpc')
        chi_star = cmb.Chistar(simulation.redshift)
        k2d = kineticsz.utils.k2D(simulation.boxsize, pixels)
        noiseck = numpy.load('/home/ugiri/kineticsz/data/Ck_sw_0.5_beam_1_in_mpc_out_mpc_hr.npz')
        el = noiseck['k']*chi_star
        Cl = (noiseck['k']**2/el**2)*(noiseck['noiseck'] + noiseck['cmbck'])
        Clcmb = Cl*(chi_star*simulation.boxsize)**2
        std2D = numpy.sqrt(numpy.interp(k2d.flatten(), el/chi_star, Clcmb/2)).reshape(k2d.shape)
        map2D = numpy.random.normal(scale=std2D) + 1j*numpy.random.normal(scale=std2D)
        map2D[int(pixels/2 +1):, 0] = numpy.conj(map2D[1:int(pixels/2),0][::-1])
            
        return (simulation.voxels/simulation.boxsize)**2*numpy.fft.irfft2(map2D)

    f = cmb.model.scale_independent_growth_factor_f(simulation.redshift)
    a = 1./(1.+simulation.redshift)
    H = cmb.model.Hubble(simulation.redshift)*cmb.c

    kFreq = 2 * numpy.pi * numpy.fft.fftfreq(simulation.voxels, simulation.boxsize/simulation.voxels)
    kx, ky, kz = numpy.meshgrid(kFreq, kFreq, kFreq[:int(simulation.voxels/2 + 1)])
    knorm = numpy.sqrt(kx**2 + ky**2 + kz**2)
    knorm[0,0,0] = 0.0001
    kx = 0; ky = 0;

    def simulate():
        x = numpy.random.normal(size=(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.float32)
        y = numpy.random.normal(size=(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.float32)
        z = numpy.random.normal(size=(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.float32)
        rnoise = numpy.array([x.ravel(), y.ravel(), z.ravel()]).astype(numpy.float32)/numpy.sqrt(2*simulation.boxsize**3)
        x = numpy.random.normal(size=(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.float32)
        y = numpy.random.normal(size=(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.float32)
        z = numpy.random.normal(size=(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.float32)
        inoise = numpy.array([x.ravel(), y.ravel(), z.ravel()]).astype(numpy.float32)/numpy.sqrt(2*simulation.boxsize**3)

        pnl = cmb.interpolate_nonlinear_powerspectrum(redshift=simulation.redshift)
        handle = numpy.load('/home/ugiri/kineticsz/data/LSS_filters.npz')
        Pgg = interpolate.interp1d(handle['k'][1:], handle['phh'][1:], bounds_error=False, fill_value=0)(knorm)
        Pmg = interpolate.interp1d(handle['k'][1:], handle['pmh'][1:], bounds_error=False, fill_value=0)(knorm)
        Pmm = pnl(knorm)
        Pmv = (f*a*H) * Pmm
        Pgv = (f*a*H) * Pmg
        Pvv = (f*a*H)**2 * Pmm
        Pvv[kz==0] = 1e-10
        Pgv[kz==0] = 1e-10
        Pmv[kz==0] = 1e-10
        dec = numpy.swapaxes(numpy.array([[Pmm.ravel() + 1 , Pmg.ravel(), Pmv.ravel()],
                                      [Pmg.ravel(), Pgg.ravel() + 50, Pgv.ravel()],
                                      [Pmv.ravel().conjugate(), Pgv.ravel().conjugate(), Pvv.ravel() +150]]), 0,2)
        
        L = numpy.linalg.cholesky(dec)
        x = L[:,0,0]*(rnoise[0,:] + 1j*inoise[0,:]) + L[:,0,1]*(rnoise[1,:]+ 1j*inoise[1,:])+ L[:,0,2]*(rnoise[2,:]+ 1j*inoise[2,:]) 
        y = L[:,1,0]*(rnoise[0,:] + 1j*inoise[0,:]) + L[:,1,1]*(rnoise[1,:]+ 1j*inoise[1,:])+ L[:,1,2]*(rnoise[2,:]+ 1j*inoise[2,:]) 
        z = L[:,2,0]*(rnoise[0,:] + 1j*inoise[0,:]) + L[:,2,1]*(rnoise[1,:]+ 1j*inoise[1,:])+ L[:,2,2]*(rnoise[2,:]+ 1j*inoise[2,:]) 
        x= utensils.hermitianize(x.reshape(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.complex64)
        y= utensils.hermitianize(y.reshape(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.complex64)
        z= utensils.hermitianize(z.reshape(simulation.voxels, simulation.voxels, int(simulation.voxels/2+1))).astype(numpy.complex64)

        return x,y,z

    def realization():
        
        noise_map = Noisemap()
        deltam, deltag, deltav = simulate()
        momentum = numpy.fft.rfftn(numpy.fft.irfftn(-1j*deltav*kz/knorm**2)*(1 + backward**3*numpy.fft.irfftn(deltam)*simulation.boxsize**3))

        return (noise_map, deltam, deltag, deltav, momentum)

    _, _, _, _, momentum = realization()
    mesh = ArrayMesh(momentum, BoxSize=simulation.boxsize)

    pkksz = Powerspectrum(ProjectedFFTPower(mesh))
    pkksz.powerspectrum *= simulation.boxsize**2

    k2d = kineticsz.utils.k2D(simulation.boxsize, simulation.voxels).astype(numpy.float32)
    k3d = kineticsz.utils.k3D(simulation.boxsize, simulation.voxels).astype(numpy.float32)

    phh = numpy.load('/home/ugiri/kineticsz/data/LSS_filters.npz')
    phh = interpolate.interp1d(phh['k'][1:-1], phh['phh'][1:-1], bounds_error=False, fill_value='extrapolate')
    pmh = numpy.load('/home/ugiri/kineticsz/data/LSS_filters.npz')
    pmh = interpolate.interp1d(pmh['k'][1:-1], pmh['pmh'][1:-1], bounds_error=False, fill_value='extrapolate')

    Phh = numpy.interp(k3d.flatten(), k3d.flatten(), phh(k3d.flatten())).reshape(k3d.shape).astype(numpy.float32)
    Pmh = numpy.interp(k3d.flatten(), k3d.flatten(), pmh(k3d.flatten())).reshape(k3d.shape).astype(numpy.float32)

    noisepk = numpy.load('/home/ugiri/kineticsz/data/Ck_sw_0.5_beam_1_in_mpc_out_mpc_hr.npz')
    Ckksz = numpy.interp(k2d.flatten(), pkksz.k.real[1:-1], (kstar**2*pkksz.powerspectrum[1:-1].real)).reshape(k2d.shape)
    Cknoise = numpy.interp(k2d.flatten(), noisepk['k'], (noisepk['cmbck'] + noisepk['noiseck'])).reshape(k2d.shape)
    Cktot = Ckksz + Cknoise

    noise_n0 = 1./(kstar**2*(forward**3*scipy.fft.rfftn(backward**3*scipy.fft.irfftn((Pmh**2/Phh), workers=40) * 
                        backward**2*scipy.fft.irfft2(1/Cktot)[:,:,None], workers=40)))
    noise_n0 = numpy.array(noise_n0, dtype=numpy.complex64)

    def bias():
        simnoise = []
        logger.info('Starting monte carlo over {} simulations'.format(simulations))
        for ii in range(simulations):
            deltam1, deltam2, deltav1 = simulate()
            deltam2, deltag2, deltav2 = simulate()
            deltam3, deltag3, deltav3 = simulate()
            
            if noise == 'N0':
                deltam3 = deltam2 = deltam1 
                deltag3 = deltag2
            else:
                assert noise.lower() == 'N1'.lower(), 'Allowed options for noise are `N1` and `N0`'

            momentum = numpy.fft.irfftn(deltav1)*(numpy.fft.irfftn(deltam3))
            T = backward**2*scipy.fft.irfft2(forward**2*scipy.fft.rfft2(((cmb.Kksz(2)*momentum.sum(2))))/Cktot)
            D = backward**3*scipy.fft.irfftn(forward**3*deltag2*(Pmh/Phh))
            v1 = T[:,:,None]*D
            momentum = numpy.fft.irfftn(deltav1)*(numpy.fft.irfftn(deltam2))
            T = backward**2*scipy.fft.irfft2(forward**2*scipy.fft.rfft2(((cmb.Kksz(2)*momentum.sum(2))))/Cktot)
            D = backward**3*scipy.fft.irfftn(forward**3*deltag3*(Pmh/Phh))
            v2 = T[:,:,None]*D

            v1 = simulation.boxsize**3*numpy.float32(backward**3*scipy.fft.irfftn(kstar * noise_n0 * forward**3*scipy.fft.rfftn(v1)))
            v2 = simulation.boxsize**3*numpy.float32(backward**3*scipy.fft.irfftn(kstar * noise_n0 * forward**3*scipy.fft.rfftn(v2).conjugate()))
            
            noisepk = FFTPower(ArrayMesh(v1/(simulation.boxsize/simulation.voxels)**3, BoxSize=simulation.boxsize), 
                        second=ArrayMesh(v2/(simulation.boxsize/simulation.voxels)**3, BoxSize=simulation.boxsize), mode='1d')
            simnoise.append((noisepk.power['power'][1:].real))
            numpy.savez(f'../data/noise_{noise}', k=noisepk.power['k'], pk=numpy.array(noisepk))
            logger.info('Simulation {} done'.format(ii+1))

    bias()

if '__main__' == __name__:
    fire.Fire(main)
