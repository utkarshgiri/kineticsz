import h5py
import numpy
import logging
import utensils
import kineticsz
from rich import print
from rich.logging import RichHandler
from matplotlib import pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

def plot_ellipse(F, ls='-', label=None, color='k'):
    
    sigmax = numpy.round(numpy.sqrt(numpy.linalg.inv(F))[1,1],2).real
    sigmay = numpy.round(numpy.sqrt(numpy.linalg.inv(F))[0,0],2).real
    sigmaxy = numpy.round((numpy.linalg.inv(F))[1,0],2).real

    width = numpy.sqrt((sigmax**2 + sigmay**2)/2 + numpy.sqrt((sigmax**2 - sigmay**2)**2/4 + sigmaxy**2))
    height = numpy.sqrt((sigmax**2 + sigmay**2)/2 - numpy.sqrt((sigmax**2 - sigmay**2)**2/4 + sigmaxy**2))
    angle = numpy.degrees(0.5*numpy.arctan(2*sigmaxy/(sigmax**2 - sigmay**2)))

    from matplotlib.patches import Ellipse

    ell = Ellipse(xy=(0, 3.24), width=2.48*width, height=2.48*height, angle=angle, ls=ls, label=label, color=color, fill=False)
    return ell

fnl, bg, bv = 50, 3.24, 1 #fiducial model parameter
beta = 2*1.42*(bg-1)

modes = numpy.load('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/2/snapdir_001/modes.npz')
kz, k, hm, mm = modes['kz'][:58], modes['k'][:58], modes['hm'][:58], modes['dm'][:58]

cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_quijote.json')
pmm = cmb.interpolate_matter_powerspectrum()(k)
alpha = cmb.interpolate_transfer_function()(k)

def analyticHaloFisher(shot_noise=4000):

    Ffnlfnl, Ffnlb, Fbb = 0, 0, 0

    for i in range(0, len(k)):
        cov= bg**2*pmm[i] + shot_noise
        cinv = 1./cov
        dcdb = (2*bg)*pmm[i]
        dcdfnl = (2*bg*beta/alpha[i])*pmm[i]
        Ffnlfnl += 0.5*cinv**2*dcdfnl**2
        Fbb += 0.5*cinv**2*dcdb**2
        Ffnlb += 0.5*cinv**2*dcdfnl*dcdb

    return numpy.array([[Fbb, Ffnlb], [Ffnlb, Ffnlfnl]])

Fhh_analytic = analyticHaloFisher()
logger.info('Analytic model based error on fnl using only Phh : {}'.format(numpy.round(numpy.sqrt(numpy.linalg.inv(Fhh_analytic))[1,1],2)))

phh = (hm*hm.conjugate()).real
pmm = (mm*mm.conjugate()).real

fnl, bg = 0, numpy.sqrt(phh[1]/pmm[1]).real
beta = 2*1.42*(bg-1)

def simulationHaloFisher(shot_noise=4000):

    Ffnlfnl, Ffnlb, Fbb = 0, 0, 0

    for i in range(1, len(k)):
        cov= phh[i].real
        cinv = 1./cov
        dcdb = (2*bg)*pmm[i]
        dcdfnl = (2*bg*beta/alpha[i])*pmm[i]
        Ffnlfnl += 0.5*cinv**2*dcdfnl**2
        Fbb += 0.5*cinv**2*dcdb**2
        Ffnlb += 0.5*cinv**2*dcdfnl*dcdb

    return numpy.array([[Fbb, Ffnlb], [Ffnlb, Ffnlfnl]])

Fhh_simulation = simulationHaloFisher()
logger.info('Simulation based error on fnl using Phh : {}'.format(numpy.round(numpy.sqrt(numpy.linalg.inv(Fhh_simulation))[1,1],2)))

def analyticDensityFisher(shot_noise=4000):

    Ffnlfnl, Ffnlbg, Fbgbg = 0, 0, 0
    covariance = numpy.zeros((2,2), dtype=numpy.float64);

    for i in range(0, len(k)):
        covariance[0,0] = (bg + fnl*beta/alpha[i])**2 * pmm[i] + shot_noise
        covariance[1,1] = pmm[i]
        covariance[0,1] = (bg + fnl*beta/alpha[i]) * pmm[i]
        covariance[1,0] = (bg + fnl*beta/alpha[i]) * pmm[i]

        cinv = numpy.linalg.inv(covariance)

        dCdbg = numpy.array([[(2*bg)*pmm[i], pmm[i]], [pmm[i], 0]])
        dCdbg = numpy.array([[(2*bg)*pmm[i], pmm[i]], [pmm[i], 0]])
        dCdfnl = numpy.array([[(2*bg*beta/alpha[i])*pmm[i], pmm[i]*beta/alpha[i]], [pmm[i]*beta/alpha[i], 0]])

        Ffnlfnl += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdfnl), cinv), dCdfnl))
        Fbgbg += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdbg), cinv), dCdbg))
        Ffnlbg += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdfnl), cinv), dCdbg))

    return numpy.array([[Fbgbg, Ffnlbg], [Ffnlbg, Ffnlfnl]])


Fmm = analyticDensityFisher()
logger.info('Error on fnl using halo and density field: {}'.format((numpy.sqrt(numpy.linalg.inv(Fmm))[1,1])))
logger.info('Error on bg using halo and density field: {}'.format((numpy.sqrt(numpy.linalg.inv(Fmm))[0,0])))

try:
    density_combined = h5py.File('/gpfs/ugiri/mcmc_chains/combined_density_quijote_58_modes.h5', 'r')['mcmc']['chain'][:].reshape(-1,2)
    logger.info('Scaled 1-sigma error from combined mcmc chians: {}'.format(numpy.std(density_combined[-1000:,1])*10))
    logger.info('Scaled 1-sigma error from combined mcmc chians: {}'.format(numpy.std(density_combined[-1000:,0])*10))
except:
    pass
f = cmb.model.scale_independent_growth_factor_f(2)
a = 1./(1.+2)
H = cmb.model.Hubble(2)*cmb.c
bv = 1.0
def analyticVelocityFisher(shot_noise=4000):

    Ffnlfnl, Fbvbv, Fbgbg, Ffnlbg, Ffnlbv, Fbgbv = 0, 0, 0, 0, 0, 0
    covariance = numpy.zeros((2,2), dtype=numpy.complex128);

    for i in range(0, len(k), 1):
        halo_prefactor = (bg + fnl*beta/alpha[i])
        velocity_prefactor = (1j*bv*f*a*H*kz[i]/k[i]**2)

        covariance[0,0] = halo_prefactor**2 * pmm[i] + shot_noise
        covariance[1,1] = velocity_prefactor * velocity_prefactor.conjugate() * pmm[i] + 2.5e9
        covariance[0,1] = halo_prefactor * velocity_prefactor * pmm[i]
        covariance[1,0] = halo_prefactor * velocity_prefactor.conjugate() * pmm[i]

        cinv = numpy.linalg.inv(covariance)

        dCdbg = numpy.array([[(2*bg)*pmm[i], velocity_prefactor*pmm[i]], [velocity_prefactor.conjugate()*pmm[i], 0]])
        dCdfnl = numpy.array([[(2*bg*beta/alpha[i])*pmm[i], pmm[i]*velocity_prefactor*beta/alpha[i]], [pmm[i]*velocity_prefactor.conjugate()*beta/alpha[i], 0]])
        dCdbv = numpy.array([[0, covariance[0,1]/bv], [covariance[1,0]/bv, (velocity_prefactor * velocity_prefactor.conjugate()*2/bv)*pmm[i]]])

        Ffnlfnl += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdfnl), cinv), dCdfnl))
        Fbvbv += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdbv), cinv), dCdbv))
        Fbgbg += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdbg), cinv), dCdbg))
        
        Ffnlbg += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdfnl), cinv), dCdbg))
        Ffnlbv += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdfnl), cinv), dCdbv))
        Fbgbv += 0.5*numpy.trace(numpy.matmul(numpy.matmul(numpy.matmul(cinv, dCdbg), cinv), dCdbv))
        
    F = numpy.array([[Fbgbg, Fbgbv, Ffnlbg], [Fbgbv, Fbvbv, Ffnlbv], [Ffnlbg, Ffnlbv, Ffnlfnl]])

    return F.real

Fvv = analyticVelocityFisher()
logger.info('Error on fnl using halo and velocity field: {}'.format((numpy.sqrt(numpy.linalg.inv(Fvv))[2,2])))
logger.info('Error on bg using halo and velocity field: {}'.format((numpy.sqrt(numpy.linalg.inv(Fvv))[0,0])))

try:
    velocity_combined = h5py.File('/home/ugiri/kineticsz/mcmc_chains/combined_velocity_quijote_58.h5', 'r')['mcmc']['chain'][:].reshape(-1,3)
    logger.info('Scaled 1-sigma error from combined mcmc chians: {}'.format(numpy.std(velocity_combined[-10000:,-1])*10))
    logger.info('Scaled 1-sigma error from combined mcmc chians: {}'.format(numpy.std(velocity_combined[-10000:,0])*10))
except:
    pass

fig, ax = plt.subplots(dpi=200, figsize=(4,4))
ax.add_patch(plot_ellipse(Fvv[[0,2],:][:,[0,2]], ls='-', label=r'$[\delta_h, v_r]$', color='r'))
ax.add_patch(plot_ellipse(Fmm,  ls='--', label=r'$[\delta_h, \delta_m]$', color='b'))
ax.add_patch(plot_ellipse(Fhh_analytic, ls='-.', label=r'$[\delta_h]$', color='k'))
ax.autoscale()
ax.legend(frameon=False, loc=2)
utensils.save_and_upload_plot(filename='fisher_ellipses.pdf', subdirectory='ksz_mcmc');