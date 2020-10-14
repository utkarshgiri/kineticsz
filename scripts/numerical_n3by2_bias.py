import numpy
import scipy
import joblib
import pathlib
import logging
import kineticsz
from decimal import Decimal
from scipy import interpolate
from scipy.integrate import simps
from rich.logging import RichHandler

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

snapshot = kineticsz.Snapshot(snapshot=list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('99/snapdir_001/*.hdf5')),
                              in_units='kpch', voxels=1024, lazy=True)

cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_params.json', units='mpc')

chistar = chistar = cmb.Chistar(redshift=snapshot.redshift)
kstar = cmb.Kksz(redshift=snapshot.redshift)

dirnames = sorted(list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/matter.powerspectrum')), key=lambda x:x.stat().st_mtime)
dirnames = [x.parent for x in dirnames]

k = joblib.load(dirnames[0].joinpath('matter-halo.powerspectrum')).k
Pge, Pvv, Pgg, Pnn, Pmm = [numpy.zeros((len(dirnames), len(k))) for _ in range(5)]

for i, name in enumerate(dirnames):
    Pge[i,:] = joblib.load(name.joinpath('matter-halo.powerspectrum')).powerspectrum
    Pvv[i,:] = joblib.load(name.joinpath('momentum.powerspectrum')).powerspectrum
    Pgg[i,:] = joblib.load(name.joinpath('halo.powerspectrum')).powerspectrum
    Pnn[i,:] = joblib.load(name.joinpath('noise.powerspectrum')).powerspectrum
    Pmm[i,:] = joblib.load(name.joinpath('matter.powerspectrum')).powerspectrum

Pge = interpolate.interp1d(k, Pge.mean(0), fill_value='extrapolate')
Pvv = interpolate.interp1d(k, Pvv.mean(0), fill_value='extrapolate')
Pgg = interpolate.interp1d(k, Pgg.mean(0), fill_value='extrapolate')
Pmm = interpolate.interp1d(k, Pmm.mean(0), fill_value='extrapolate')

Cltot = numpy.load('/home/ugiri/kineticsz/data/analytic_noise.npz')
Cltot = interpolate.interp1d(Cltot['l'], Cltot['cl'], fill_value='extrapolate')

# Integration in polar coordinates k-volume space is smaller in polar than 3D grid
# kL = 0 limit

f = cmb.model.scale_independent_growth_factor_f(snapshot.redshift)
a = 1./(1.+snapshot.redshift)
H = cmb.model.Hubble(snapshot.redshift)*cmb.c
bv = 1.0

k = numpy.linspace(snapshot.fundamental_mode, snapshot.nyquist, 1000)[:]
Pmm = cmb.interpolate_linear_powerspectrum()
Pvr = interpolate.interp1d(k, (f*a*H/k)**2*Pmm(k)/3., fill_value=0, bounds_error=False)

bg = 3.24; shot_noise=3920; halo_variance_by_mean = 2.3

def integrate_B(k):
    return (4*numpy.pi*k**2*Pvr(k)*(bg)**2*Pmm(k))/(2*numpy.pi)**3

n3by2B = simps(integrate_B(k), k)
logger.info('N3/2 B term @ kL=0: %.2E' % Decimal(n3by2B))


logger.info('Integrating using dblquad...')
bins = 40
bg=3.24
def integrate(q, qdash):
    #return ((f*a*H)**2*bg**2/(24*numpy.pi*numpy.pi))*(q**2+qdash**2)*(Pmm(q)/q)*(Pmm(qdash)/qdash)
    return ((f*a*H)**2*bg**2/(24*numpy.pi*numpy.pi))*(1)*(Pmm(q)/q)*(Pmm(qdash)/qdash)
    

N3by2kL = numpy.zeros(bins)

k = numpy.concatenate((numpy.geomspace(0.002, 0.02, 15), numpy.geomspace(0.02, 0.1, 12), numpy.geomspace(0.1, 1, 10), numpy.geomspace(1,2,5)))
for i,kl in enumerate(k):
    ans = (scipy.integrate.dblquad(integrate, snapshot.fundamental_mode, snapshot.nyquist, lambda a: numpy.abs(a-kl), lambda b: numpy.abs(b+kl)))[0]*kl
    N3by2kL[i] = ans
    numpy.savez('../data/n3by2-bias-dblquad', k=k, n3by2=N3by2kL)

logger.info('Calculation over. Data saved in /home/ugiri/kineticsz/data/n3by2-bias-dblquad')
