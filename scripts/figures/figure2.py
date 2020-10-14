import numpy
import joblib
import pathlib
import utensils
import kineticsz
from  classylss import binding
from scipy import interpolate
from matplotlib import pyplot as plt

from scipy.integrate import simps
plt.style.use('science')

snapshot = kineticsz.Snapshot(snapshot=list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('99/snapdir_001/*.hdf5')),
                              in_units='kpch', voxels=1024, lazy=True)
kint = numpy.linspace(snapshot.fundamental_mode, snapshot.nyquist, snapshot.voxels)
epsilon = 1e-6
muint = numpy.linspace(-1+epsilon,1-epsilon, snapshot.voxels, endpoint=False)
cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_quijote.json', units='mpc')

filenames = list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/matter.powerspectrum'))
print('Nume files : ', len(filenames))
pkmm, kk = [], []
for name in filenames:
    pkmm.append(joblib.load(name).powerspectrum)
    kk = joblib.load(name).k
pmm_nl = interpolate.interp1d(kk, numpy.array(pkmm).mean(0), bounds_error=False, fill_value='extrapolate')
pmm_lin = cmb.interpolate_linear_powerspectrum(redshift=2)
pmm_nl = cmb.interpolate_nonlinear_powerspectrum(redshift=2)

classy = binding.ClassEngine(cmb.parameters)
background = binding.Background(classy)
f = background.scale_independent_growth_rate(cmb.redshift)
h = background.hubble_function(cmb.redshift)*cmb.c
a = 1./(1. + snapshot.redshift)
factor = f * a * h

Ik = numpy.zeros_like(kint)
for i,k in enumerate(kint):

    def F(mu, kp):
        return (kp**2/((4*numpy.pi**2))*pmm_nl(numpy.sqrt((k**2+kp**2-2*kp*k*mu)))*pmm_nl(kp)*((k*(k-2*kp*mu)*(1-mu**2))/(kp**2*(k**2+kp**2-2*kp*k*mu))))
    
    z = F(muint.reshape(-1,1), kint.reshape(1,-1))
    Ik[i] = factor**2*simps([simps(zmu, muint) for zmu in z], kint)

dirnames = list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/ksz.powerspectrum'))
pkksz = []
for name in dirnames[:]:
    pkksz.append(joblib.load(name).powerspectrum)

simksz_k = joblib.load(dirnames[0]).k
simksz_powerspectrum = numpy.array(pkksz).mean(0)

inp = interpolate.interp1d(simksz_k, simksz_powerspectrum, fill_value='extrapolate')

kstar = cmb.Kksz(snapshot.redshift)
chistar = cmb.Chistar(snapshot.redshift)
cmbcl = cmb.Cl(lmax=int(snapshot.nyquist*chistar), scale_invariant=True)

kszk = simksz_k*chistar
kintegral = kint*chistar
noise = (0.5*numpy.pi/180./60.)**2*numpy.exp((cmbcl.ell*(cmbcl.ell+1)*(1*numpy.pi/180./60.)**2)/8*numpy.log(2))/(2*numpy.pi)

dirnames = list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/'))
pkmh, pkhh, k = [], [], []
for name in dirnames:
    try:
        pmh = joblib.load(name.joinpath('matter-halo.powerspectrum'))
        phh = joblib.load(name.joinpath('halo.powerspectrum'))
        pkmh.append(pmh.powerspectrum)
        pkhh.append(phh.powerspectrum)
        k = pmh.k
    except:
        continue

fig, ax = plt.subplots(nrows=2, dpi=500, figsize=(6,6))
end = 256; ls=(0, (5, 1)); lw=2
sl = 500


ax[0].semilogy(k[k>sl/chistar], numpy.array(pkmh).mean(0)[k>sl/chistar], label=r'$P_{mh}$', lw=2, ls=ls)
ax[0].semilogy(k[k>sl/chistar], numpy.array(pkhh).mean(0)[k>sl/chistar], label=r'$P_{hh}$', lw=2, ls=(0, (3, 1, 1, 1)))
ax[0].legend(frameon=False)
ax[0].set_ylabel('${P(k) \ [\mathrm{Mpc}^3]}$', fontsize=12)
ax[0].set_xlabel('${k \ [\mathrm{Mpc}^{-1}]}$', fontsize=12)
ax[0].grid(ls='--', alpha=0.5)
cmbck = numpy.interp(kint*chistar, cmbcl.ell[cmbcl.ell>sl], (2*numpy.pi)*cmbcl.cl[cmbcl.ell>sl]/cmbcl.ell[cmbcl.ell>sl]**2)
noiseck = numpy.interp(kint*chistar, cmbcl.ell[cmbcl.ell>sl], (2*numpy.pi)*noise[cmbcl.ell>sl])
kszck = numpy.interp(kint*chistar, kintegral[kintegral>sl], kstar**2*kint[kintegral>sl]**2*snapshot.boxsize*numpy.array(Ik)[kintegral>sl]/2/kintegral[kintegral>sl]**2)                  

ax[1].semilogy(kszk[kszk>sl], kstar**2*simksz_k[kszk>sl]**2*simksz_powerspectrum[kszk>sl]/(2*numpy.pi), lw=2, label='$C_l^{kSZ}$', ls=ls)
ax[1].semilogy(cmbcl.ell[cmbcl.ell>sl], cmbcl.cl[cmbcl.ell>sl],  lw=2, label='$C_l^{cmb}$', ls=(0, (3, 1, 1, 1)))
ax[1].semilogy(cmbcl.ell[cmbcl.ell>sl], cmbcl.ell[cmbcl.ell>sl]*(cmbcl.ell[cmbcl.ell>sl]+1)*noise[cmbcl.ell>sl], lw=2, label='$C_l^{noise}$', ls='dashdot')
ax[1].semilogy(kintegral[kintegral>sl], kstar**2*kint[kintegral>sl]**2*snapshot.boxsize*numpy.array(Ik)[kintegral>sl]/2/(2*numpy.pi),  lw=2, label='$C_l^{SkSZ}$', ls=(0, (3, 1, 1, 1, 1, 1)))

ax[1].grid(ls='--', alpha=0.5)
ax[1].set_xlabel('${l}$', fontsize=12);
ax[1].set_ylabel('${\\frac{l(l+1)}{2\pi}C_l \ [\\mu \mathrm{K} ^2]}$', fontsize=12);
ax[1].legend(frameon=False);
ax[1].tick_params(axis='x', labelsize=12)
ax[1].tick_params(axis='y', labelsize=12)
plt.tight_layout()
utensils.save_and_upload_plot(filename='figures/figure2.pdf')

numpy.savez('/home/ugiri/kineticsz/data/LSS_filters.npz', k=k, pmh=numpy.array(pkmh).mean(0), phh=numpy.array(pkhh).mean(0))
