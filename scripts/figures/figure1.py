import numpy
import joblib
import pathlib
import utensils
from kineticsz import pycmb
from matplotlib import gridspec
from matplotlib import pyplot as plt
plt.style.use('science')

dirname = pathlib.Path('/gpfs/ugiri/gadget/size1024')
shotnoise = numpy.load('/home/ugiri/kineticsz/data/shotnoise.npy')
residuals = numpy.load('/home/ugiri/kineticsz/data/residuals.npz')

bias_minus50, bias_plus50 = numpy.zeros((4, 511)), numpy.zeros((4, 511))

for i,basename in enumerate(['fnlminus50z127n1', 'fnlminus50z127n2', 'fnlminus50z127n3', 'fnlminus50z127n5']):
    k = joblib.load(dirname.joinpath(basename, 'matter-halo.powerspectrum')).k[1:]
    pmh = joblib.load(dirname.joinpath(basename, 'matter-halo.powerspectrum')).powerspectrum
    pmm = joblib.load(dirname.joinpath(basename, 'matter.powerspectrum')).powerspectrum
    bias_minus50[i,:] = pmh[1:]/pmm[1:]

for i,basename in enumerate(['fnl50z127n1', 'fnl50z127n2', 'fnlplus50z127n4', 'fnlplus50z127n5']):
    pmh = joblib.load(dirname.joinpath(basename, 'matter-halo.powerspectrum')).powerspectrum
    pmm = joblib.load(dirname.joinpath(basename, 'matter.powerspectrum')).powerspectrum
    bias_plus50[i,:] = pmh[1:]/pmm[1:]

bias_plus50_error = numpy.std(numpy.array(bias_plus50), axis=0)
bias_minus50_error = numpy.std(numpy.array(bias_plus50), axis=0)

cmb = pycmb.CMB('/home/ugiri/kineticsz/ics/configuration/class_quijote.json')
alpha = cmb.interpolate_transfer_function()

#best-fit parameters from mcmc
plus50 =  [3.15 ,  0.973 , 50.2]
minus50 = [3.01, 0.99, -55]

bias_model_fnlplus50 = ((plus50[0] + (2*1.42*(plus50[0]-1)*(plus50[-1]/plus50[1]**2))/alpha(k)))
bias_model_fnlminus50 = ((minus50[0] + (2*1.42*(minus50[0]-1)*(minus50[-1]/minus50[1]**2))/alpha(k)))

rk = residuals['k'][1:]
cmap = utensils.palletes.plotly_color
fig = plt.figure(dpi=200, figsize=(6,4))
gs = gs = gridspec.GridSpec(36,36)
ax0 = fig.add_subplot(gs[:16, :])
ax1 = fig.add_subplot(gs[20:, :], sharex=ax0)
ls=(0, (5, 1)); lw=2

ax1.loglog(rk[rk<0.1], numpy.mean(residuals['pk'], axis=0)[1:][rk<0.1], label=r'${P_{\delta_{h}^{\prime}}}$', color=cmap[0], ls=ls, lw=2)
ax1.loglog(rk[rk<0.1], (numpy.ones_like(residuals['k'][1:])*numpy.mean(shotnoise))[rk<0.1], label=r'${\frac{1}{n_h}}$', color=cmap[1], lw=2, ls='-.')
ax0.grid(ls='--', alpha=0.2)
ax1.grid(ls='--', alpha=0.2)

ax0.errorbar(k[k<0.1], numpy.mean(numpy.array(bias_minus50), axis=0)[k<0.1], yerr=numpy.std(numpy.array(bias_minus50), axis=0)[k<0.1], marker='s', lw=2, ms=4, ls='none',  color=cmap[1], label=r'${f_{NL}=+50}$')
ax0.semilogx(k[k<0.1], bias_model_fnlminus50[k<0.1],  color = cmap[1], lw=2, ls='-.')

ax0.axvline(0.012, color='black', lw=1, alpha=0.3, ls='--')
ax0.errorbar(k[k<0.1], numpy.mean(numpy.array(bias_plus50), axis=0)[k<0.1], yerr=numpy.std(numpy.array(bias_plus50), axis=0)[k<0.1], marker='o', lw=2, ms=4, ls = 'none',  color=cmap[0], label=r'${f_{NL}=-50}$')
ax0.semilogx(k[k<0.1], bias_model_fnlplus50[k<0.1], color=cmap[0], lw=2, ls='--')
ax1.set_ylim(bottom = 0.8e3, top=1.5e4)
ax0.set_ylim(bottom = 0, top=6)
ax1.set_xlabel(r'${k \ [\mathrm{Mpc}^{-1}]}$', fontsize=12)
ax1.set_ylabel(r'${P(k, z=2) \ [\mathrm{Mpc}^{3}]}$', fontsize=12)
ax0.set_ylabel(r'${\frac{P_{mh}}{P_{mm}}}$', fontsize=15)
ax1.legend(frameon=False, fontsize=10)
ax0.legend(frameon=False, fontsize=10)

plt.tight_layout()
utensils.save_and_upload_plot('figures/figure1.pdf')
