import numpy
import scipy
import joblib
import pathlib
import logging
import kineticsz
from numpy import sin, cos
from decimal import Decimal
from scipy import interpolate
from scipy.integrate import simps
from rich.logging import RichHandler

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

snapshot = kineticsz.Snapshot(snapshot=list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('99/snapdir_001/*.hdf5')),
                              in_units='kpch', voxels=1024, lazy=True)

cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_quijote.json', units='mpc')

chistar = chistar = cmb.Chistar(snapshot.redshift)
kstar = cmb.Kksz(snapshot.redshift)

dirnames = sorted(list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/matter.powerspectrum')), key=lambda x:x.stat().st_mtime)
dirnames = [x.parent for x in dirnames]
k = joblib.load(dirnames[0].joinpath('matter-halo.powerspectrum')).k
Pge, Pvv, Pgg, Pnn, Pksz = [numpy.zeros((len(dirnames), len(k))) for _ in range(5)]

for i, name in enumerate(dirnames[:]):
    Pge[i,:] = joblib.load(name.joinpath('matter-halo.powerspectrum')).powerspectrum
    Pvv[i,:] = joblib.load(name.joinpath('momentum.powerspectrum')).powerspectrum
    Pgg[i,:] = joblib.load(name.joinpath('halo.powerspectrum')).powerspectrum
    Pnn[i,:] = joblib.load(name.joinpath('noise.powerspectrum')).powerspectrum
    Pksz[i,:] = joblib.load(name.joinpath('ksz.powerspectrum')).powerspectrum

Pge = interpolate.interp1d(k, Pge.mean(0), fill_value=0, bounds_error=False)
Pvv = interpolate.interp1d(k, Pvv.mean(0), fill_value=0, bounds_error=False)
Pgg = interpolate.interp1d(k, Pgg.mean(0), fill_value=numpy.inf, bounds_error=False)
Pksz = interpolate.interp1d(k, Pksz.mean(0), fill_value=numpy.inf, bounds_error=False)

Cltot = numpy.load('/home/ugiri/kineticsz/data/analytic_noise.npz')
Cltot = interpolate.interp1d(Cltot['l'][1:], Cltot['cl'][1:], fill_value=numpy.inf, bounds_error=False)

noisepk = numpy.load('/home/ugiri/kineticsz/data/Ck_sw_0.5_beam_1_in_mpc_out_mpc_hr.npz')
Ckksz = interpolate.interp1d(noisepk['k'], (kstar**2*Pksz(noisepk['k']).real))
Cknoise = interpolate.interp1d(noisepk['k'], (noisepk['cmbck'] + noisepk['noiseck']))
Cktot = interpolate.interp1d(noisepk['k'], Ckksz(noisepk['k']) + Cknoise(noisepk['k']), fill_value=numpy.inf, bounds_error=False)

l = numpy.linspace(snapshot.fundamental_mode, snapshot.nyquist/1.0, 5000)[:]*chistar
theta = numpy.linspace(0, 2*numpy.pi, 5000)[:]

def integrate(theta, l): 
    ks = numpy.sqrt(((l*cos(theta))**2 + (l*sin(theta))**2)/chistar**2)
    return (l*Pge(ks)**2)/(Pgg(ks)*(ks**2/l**2)*Cktot(ks))/(2*numpy.pi)**2

z = integrate(theta[:,None], l[None,:])

logger.info('N0 bias from simulations %.2E'%(Pnn.mean(0)[1]))
logger.info('N0 bias from polar integration %.2E' % Decimal((chistar**4/kstar**2)/simps([simps(zmu, theta) for zmu in z], l)))

l = numpy.linspace(-1*snapshot.nyquist/1.0, snapshot.nyquist/1.0, 5000)[:]*chistar

def integrate(lx, ly): 
    l = numpy.sqrt(lx**2 + ly**2)
    ks = l/chistar
    return (Pge(ks)**2)/(Pgg(ks)*(ks**2/l**2)*Cktot(ks))/(2*numpy.pi)**2

z = integrate(l[:,None], l[None,:])
logger.info('N0 bias from simps integration %.2E' % Decimal((chistar**4/kstar**2)/simps([simps(zmu, l) for zmu in z], l)))

N0 = scipy.integrate.dblquad(integrate, -1*chistar*snapshot.nyquist/1.05, chistar*snapshot.nyquist/1.05, lambda x: -1*chistar*snapshot.nyquist/1.05, lambda x: chistar*snapshot.nyquist/1.05)

logger.info('N0 bias from dblquad integration %.2E' % Decimal((chistar**4/kstar**2)/N0[0]))
