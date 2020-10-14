import numpy
import scipy
import joblib
import utensils
import pathlib
from matplotlib import pyplot as plt
plt.style.use('science')

numerical_n1_bias = numpy.load('../../data/n1-bias-quad-angle-averaged.npz')
numerical_n0_bias = numpy.full_like(numerical_n1_bias['n1_bias'], 1.08e9) #convolution based
numerical_n3by2_bias = numpy.load('../../data/n3by2-bias-dblquad1.npz')

dirnames = sorted(list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/matter.powerspectrum')))
dirnames = [x.parent for x in dirnames]
k = joblib.load(dirnames[0].joinpath('matter-halo.powerspectrum')).k
pnn = numpy.zeros((len(dirnames),len(k)))
pn0 = numpy.zeros((len(dirnames),len(k)))
for i, name in enumerate(dirnames):
    pnn[i,:] = joblib.load(name.joinpath('residual-noise.powerspectrum')).powerspectrum
    pn0[i,:] = joblib.load(name.joinpath('noise.powerspectrum')).powerspectrum
    

contour_ls=[(0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1))]
plt.figure(dpi=200)
c = scipy.constants.c/1e3
fig, ax = plt.subplots(dpi=200, figsize=(6,4))
#ax.loglog(k[1:], numpy.interp(k[1:], numerical_n1_bias['k'], numerical_n0_bias )/9e10, label='$N^{\mathrm{(0)}}$', lw=2, ls=contour_ls[2])
ax.loglog(k[1:][k[1:]<1], pn0.mean(0)[1:][k[1:]<1]/9e10, label='$N^{\mathrm{(0)}}$', lw=2, ls=contour_ls[2])
ax.loglog(k[1:][k[1:]<1], numpy.interp(k[1:][k[1:]<1], numerical_n1_bias['k'], numerical_n1_bias['n1_bias']), label='$N^{\mathrm{(1)}}$', lw=2, ls=contour_ls[1])
#ax.loglog(numerical_n1_bias['k'], numerical_n1_bias['n1_bias'], label='$N^{\mathrm{(1)}}$', lw=2, ls=contour_ls[1])
#ax.loglog(numerical_n3by2_bias['k'][:40], (5.62E+08 + numerical_n3by2_bias['n3by2'])/9e10, label='$N^{\mathrm{(3/2)}}$', lw=2, ls='dashdot')
ax.loglog(k[1:][k[1:]<1], numpy.interp(k[1:][k[1:]<1], numerical_n3by2_bias['k'][:40], (5.50E+08 + numerical_n3by2_bias['n3by2'])/9e10), label='$N^{\mathrm{(3/2)}}$', lw=2, ls='dashdot')
ax.loglog(k[1:][k[1:]<1], (numpy.interp(k[1:][k[1:]<1], k, pn0.mean(0)[:])/9e10 + 
                   numpy.interp(k[1:][k[1:]<1], numerical_n1_bias['k'], numerical_n1_bias['n1_bias']) +
                   numpy.interp(k[1:][k[1:]<1], numerical_n3by2_bias['k'][:40], (5.50E+08 + numerical_n3by2_bias['n3by2'])/9e10)),
         label='$\mathrm{Total} \ (N^{\mathrm{(0)}} + N^{\mathrm{(1)}} + N^{\mathrm{(3/2)}})$', lw=2,  ls=contour_ls[0])
ax.loglog(k[1:][k[1:]<1], pnn.mean(0)[1:][k[1:]<1]/9e10,  lw=2, label='$\mathrm{Simulation}$', ls='dashed')

ax.legend(frameon=False, fontsize=12)
ax.grid(which='both',ls='-.', alpha=0.5)
ax.set_ylabel(r'${P(k) \ [\mathrm{Mpc}^{3}]}$')
ax.set_xlabel(r'${k \ [\mathrm{Mpc}^{-1}]}$');
plt.tight_layout()
utensils.save_and_upload_plot(filename='figures/figure6.pdf', dpi=500)

