import numpy
import joblib
import pathlib
import utensils
import kineticsz
import matplotlib
from matplotlib import pyplot as plt

plt.style.use('science')

dirnames = sorted(list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('[1-9][0-9]/snapdir_001/modes.npz')))
snapshot = kineticsz.Snapshot(snapshot=list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/11/snapdir_001/').glob('*.hdf5')), in_units='kpch')
binsize=15
bins=numpy.geomspace(4*numpy.pi/1490.0909, 0.1, binsize)
powerspectra_r = numpy.zeros((len(dirnames), binsize-1), dtype=numpy.float64)
powerspectra_v = numpy.zeros((len(dirnames), binsize-1), dtype=numpy.float64)
powerspectra_q = numpy.zeros((len(dirnames), binsize-1), dtype=numpy.float64)
'''
for i,modefile in enumerate(dirnames[:]):

    h = numpy.load(modefile)
    k, vm, qm =  h['k'], h['vm'], h['qm']
    vm = vm[k<0.1]
    qm = qm[k<0.1]
    rm = kineticsz.Density.load(modefile.parent.joinpath('residual_noise.density')).complex
    print(rm.shape)
    kx, ky, kz, kk, rm, indices = kineticsz.utils.modes_and_indices(rm, boxsize=snapshot.boxsize)
    print(rm.shape)
    print(k.shape)
    #rm = rm.flatten()[indices]
    rm = rm[k<0.1]
    k = k[k<0.1]
    power_r = rm*rm.conjugate()
    power_v = vm*vm.conjugate()
    power_q = qm*qm.conjugate()
    weight, bins = numpy.histogram(k, bins=bins)
    powerspectrum_r, bins = numpy.histogram(k, bins=bins, weights=power_r)
    powerspectrum_v, bins = numpy.histogram(k, bins=bins, weights=power_v)
    powerspectrum_q, bins = numpy.histogram(k, bins=bins, weights=power_q)
    powerspectra_r[i,:] = powerspectrum_r[:].real/weight
    powerspectra_v[i,:] = powerspectrum_v[:].real/weight
    powerspectra_q[i,:] = powerspectrum_q[:].real/weight

powerspectra_r = powerspectra_r[~numpy.all(powerspectra_r == 0, axis=1)]
powerspectra_q = powerspectra_q[~numpy.all(powerspectra_q == 0, axis=1)]
powerspectra_v = powerspectra_v[~numpy.all(powerspectra_v == 0, axis=1)]

cov_r = numpy.corrcoef(powerspectra_r, rowvar=False)
cov_q = numpy.corrcoef(powerspectra_q, rowvar=False)
cov_v = numpy.corrcoef(powerspectra_v, rowvar=False)

numpy.savez('../../data/correlation_matrix', bins=bins, cov_vv=cov_v, cov_qq=cov_q, cov_rr=cov_r) 
'''
handle = numpy.load('../../data/correlation_matrix.npz')

bins = handle['bins']
cov_r = handle['cov_rr']
cov_q = handle['cov_qq']
cov_v = handle['cov_vv']

fig, ax = plt.subplots(nrows=1, ncols=3, dpi=500, figsize=(12,3), sharey=True)
c = ax[0].pcolormesh(cov_v.real[:,:], cmap=matplotlib.cm.Spectral, clim=(-1, 1), vmin=-1, vmax=1)
plt.sca(ax[0])
plt.xticks(numpy.arange(2,len(bins),3), numpy.round(bins,3)[2::3], fontsize=10)
plt.yticks(numpy.arange(2,len(bins),3), numpy.round(bins,3)[2::3], fontsize=10)

plt.xlabel(r'${k \ [\mathrm{Mpc}^{-1}]}$', fontsize=10)
plt.ylabel(r'${k \ [\mathrm{Mpc}^{-1}]}$', fontsize=10)
plt.title(r"$\mathrm{Corr}{(P_{\hat{v}_{r} (k)}, P_{\hat{v}_{r}(k)})}$")

c = ax[1].pcolormesh(cov_q.real[:,:], cmap=matplotlib.cm.Spectral, clim=(-1, 1), vmin=-1, vmax=1)
plt.sca(ax[1])
plt.xticks(numpy.arange(2,len(bins),3), numpy.round(bins,3)[2::3], fontsize=10)
plt.yticks(numpy.arange(2,len(bins),3), numpy.round(bins,3)[2::3], fontsize=10)


plt.xlabel(r'${k \ [\mathrm{Mpc}^{-1}]}$', fontsize=10)
plt.title(r"$\mathrm{Corr}{(P_{{q}_{r} (k)}, P_{{q}_{r}(k)})}$")

c = ax[2].pcolormesh(cov_r.real[:,:], cmap=matplotlib.cm.Spectral, clim=(-1, 1), vmin=-1, vmax=1)
plt.sca(ax[2])
plt.xticks(numpy.arange(2,len(bins),3), numpy.round(bins,3)[2::3], fontsize=10)
plt.yticks(numpy.arange(2,len(bins),3), numpy.round(bins,3)[2::3], fontsize=10)


plt.xlabel(r'${k \ [\mathrm{Mpc^{-1}]}}$', fontsize=10)
plt.title(r"$\mathrm{Corr}{(P_{{\eta}(k)}, P_{{\eta}(k)})}$")


fig.colorbar(c, ax=ax)
utensils.save_and_upload_plot('figures/figure5.pdf')
#utensils.save_and_upload_plot('correlation_matrix_with_residuals.png')

