import numpy
import scipy
import joblib
import logging
import pathlib
import kineticsz
from numpy import sin, cos
from decimal import Decimal
from scipy import interpolate
from scipy.integrate import simps
from scipy.integrate import trapz
from rich.logging import RichHandler

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


snapshot = kineticsz.Snapshot(snapshot=list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('99/snapdir_001/*.hdf5')),
                              in_units='kpch', voxels=1024, lazy=True)

cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_quijote.json', units='mpc')
c = cmb.c
chistar = chistar = cmb.Chistar(snapshot.redshift)
kstar = cmb.Kksz(snapshot.redshift)

dirnames = sorted(list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/matter.powerspectrum')))
dirnames = [x.parent for x in dirnames]
k = joblib.load(dirnames[0].joinpath('matter-halo.powerspectrum')).k
Pge, Pvv, Pgg, Pnn, Pmm, Pksz = [numpy.zeros((len(dirnames),len(k))) for _ in range(6)]

for i, name in enumerate(dirnames):
    Pge[i,:] = joblib.load(name.joinpath('matter-halo.powerspectrum')).powerspectrum
    Pvv[i,:] = joblib.load(name.joinpath('momentum.powerspectrum')).powerspectrum
    Pgg[i,:] = joblib.load(name.joinpath('halo.powerspectrum')).powerspectrum
    Pnn[i,:] = joblib.load(name.joinpath('noise.powerspectrum')).powerspectrum
    Pmm[i,:] = joblib.load(name.joinpath('matter.powerspectrum')).powerspectrum
    Pksz[i,:] = joblib.load(name.joinpath('ksz.powerspectrum')).powerspectrum

Pge = interpolate.interp1d(k, Pge.mean(0), fill_value=0, bounds_error=False)
Pvv = interpolate.interp1d(k, Pvv.mean(0), fill_value=0, bounds_error=False)
Pgg = interpolate.interp1d(k, Pgg.mean(0), fill_value=numpy.inf, bounds_error=False)
Pmm = interpolate.interp1d(k, Pmm.mean(0), fill_value=0, bounds_error=False)
Pksz = interpolate.interp1d(k, Pksz.mean(0), fill_value=numpy.inf, bounds_error=False)

Cltot = numpy.load('/home/ugiri/kineticsz/data/analytic_noise.npz')
Cltot = interpolate.interp1d(Cltot['l'], Cltot['cl'], fill_value=numpy.inf, bounds_error=False)

noisepk = numpy.load('/home/ugiri/kineticsz/data/Ck_sw_0.5_beam_1_in_mpc_out_mpc_hr.npz')
Ckksz = interpolate.interp1d(noisepk['k'], (kstar**2*Pksz(noisepk['k']).real))
Cknoise = interpolate.interp1d(noisepk['k'], (noisepk['cmbck'] + noisepk['noiseck']))
Cktot = interpolate.interp1d(noisepk['k'], Ckksz(noisepk['k']) + Cknoise(noisepk['k']), fill_value=numpy.inf, bounds_error=False)

f = cmb.model.scale_independent_growth_factor_f(snapshot.redshift)
a = 1./(1.+snapshot.redshift)
H = cmb.model.Hubble(snapshot.redshift)*cmb.c
bv = 1.0

k = numpy.geomspace(1e-4, 10, 1000)[:]
Pvr = interpolate.interp1d(k, (f*a*H/k)**2*Pmm(k)/3., fill_value=0, bounds_error=False)


l = numpy.linspace(0.005, snapshot.nyquist/1.05, 200)[1:]*chistar
theta = numpy.linspace(0, 2*numpy.pi, 200)[1:]
def integrate(t1, l1, t2, l2): 
    ks = numpy.sqrt(((l1*cos(t1))**2 + (l1*sin(t1))**2)/chistar**2)
    ksp = numpy.sqrt(((l2*cos(t2))**2 + (l2*sin(t2))**2)/chistar**2)
    q = numpy.sqrt(((l1*cos(t1) + l2*cos(t2))**2 + (l1*sin(t1) + l2*sin(t2))**2)/chistar**2) 
    try:
        Pvq = Pvr(q)
    except ValueError as e:
        print(e, q.max())
    return 1*1e18*(kstar**4/chistar**8)*((l1*Pge(ks)**2)/(Pgg(ks)*Cltot(l1))/(2*numpy.pi)**2) * ((l2*Pge(ksp)**2)/(Pgg(ksp)*Cltot(l2))/(2*numpy.pi)**2)*Pvq

z = integrate(theta[:,None, None, None], l[None,:, None, None], theta[None, None, :, None], l[None, None, None,:])

z1 = simps([simps(z0, theta) for z0 in z], l)
N1 = simps([simps(z0, theta) for z0 in z1], l)

logger.info('N1 in kL=0 limit : %.2E' % Decimal(N1/cmb.c**2))




#Angle averaged N1 bias
N0 = 1.08e9

def integrate_prefactor(lx, ly):
    l = numpy.sqrt(lx**2 + ly**2)
    ks = l/chistar
    return ((N0*kstar**2/chistar**4)*Pge(ks)**2/(Pgg(ks)*(ks**2/l**2)*Cktot(ks)))**2/((2*numpy.pi)**2)

z = integrate_prefactor(l[:,None], l[None,:])
A = simps([simps(z0, l) for z0 in z], l)

def integrateA(q):
    return 3*Pvr(q)/q

def integrateB(q):
    return 3*Pvr(q)*q**2

bins=1000
N1kLA = numpy.zeros(bins)
N1kLB = numpy.zeros(bins)
for i,kl in enumerate(numpy.linspace(snapshot.fundamental_mode/2, snapshot.nyquist/1.01, bins)):
    N1kLB[i] = (A*scipy.integrate.quad(integrateB, 0, kl)[0]*(chistar**2/(6*numpy.pi*kl)))
    N1kLA[i] = (A*scipy.integrate.quad(integrateA, kl, numpy.inf)[0]*(kl**2*chistar**2/(6*numpy.pi)))


numpy.savez('../data/n1-bias-quad-angle-averaged.npz', k=numpy.linspace(snapshot.fundamental_mode, snapshot.nyquist/1.03, bins), n1_bias=(N1kLA+N1kLB)/cmb.c**2)



l = numpy.linspace(snapshot.fundamental_mode, snapshot.nyquist/, bins)[1:]*chistar
theta = numpy.linspace(0, 2*numpy.pi, bins)[1:]
N0 = 1.08e9
def integrateA(theta, l):
    ks = l/chistar
    ones = theta/theta
    return ones*l*((N0*kstar**2/chistar**4)*Pge(ks)**2/(Pgg(ks)*Cltot(l)))**2/((2*numpy.pi)**2)

z = integrateA(theta[:,None], l[None,:])

A = simps([simps(z0, theta) for z0 in z], l)

BkLr = []
kLr = numpy.geomspace(snapshot.fundamental_mode/2, 1, bins)
for k in kLr:
    q = numpy.linspace(k, 1.1, bins)
    def integrateB(q):
        return 3*Pvr(q)/q

    z = integrateB(q)
    BkLr.append(simps(z, q))


N1kLr = A*numpy.array(BkLr)*kLr**2*chistar**2/(2*numpy.pi)
N1kLr = interpolate.interp1d(kLr, N1kLr, fill_value=0.0, bounds_error=False)

freq = 2*numpy.fft.fftfreq(snapshot.voxels, snapshot.boxsize/snapshot.voxels)
kx, ky, kz = numpy.meshgrid(freq, freq, freq[:int(snapshot.voxels/2+1)])
kmod = numpy.sqrt(kx**2 + ky**2 + kz**2)
N1kL = N1kLr(numpy.abs(kz))

weight, bins = numpy.histogram(kmod, bins=freq[:snapshot.voxels//2])
powerspectrum, bins = numpy.histogram(kmod, bins=freq[:snapshot.voxels//2], weights=N1kL)
powerspectra = powerspectrum/weight
numpy.savez('../data/n1-bias-brute-force.npz', k=bins, n1_bias=powerspectra/cmb.c**2)


