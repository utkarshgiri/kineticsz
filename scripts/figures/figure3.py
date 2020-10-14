import numpy
import joblib
import utensils
from nbodykit.lab import *
from matplotlib import pyplot as plt
plt.style.use(['science', 'ieee'])

meanr, meanb, meant, meand = [], [], [], []
freq = 2*numpy.pi*numpy.fft.fftfreq(1024, 1490.0909/1024).astype(numpy.float32)[:]
for index in range(0,100):
    try:
        pk = joblib.load(f'/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/{index}/snapdir_001/matter.powerspectrum')
        modes = numpy.load(f'/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/{index}/snapdir_001/modes.npz')
        modes_distorted = numpy.load(f'/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/{index}/snapdir_001/distorted_modes.npz')
    except:
        print('index {} failed'.format(index))
        continue
    k, kz, vm, qm, nm = (modes[x] for x in ('k', 'kz', 'vm', 'qm', 'nm'))
    vmd, qmd,= (modes_distorted[x] for x in ('vm', 'qm'))
    
    r = numpy.divide(vm*qm.conjugate(), numpy.sqrt(vm*vm.conjugate() * qm*qm.conjugate()))
    theory = numpy.divide(qm*qm.conjugate(), numpy.sqrt(qm*qm.conjugate()) * numpy.sqrt(qm*qm.conjugate() + nm*nm.conjugate()))
    bias = numpy.divide(vm*qm.conjugate(),(qm*qm.conjugate()))
    dbias = numpy.divide(vmd*qmd.conjugate(),(qmd*qmd.conjugate()))
    rval, bins = numpy.histogram(k[numpy.abs(numpy.divide(kz,k))!=0.], bins=freq[1:30][:], weights=r.real[numpy.abs(numpy.divide(kz,k))!=0.])
    bval, bins = numpy.histogram(k[numpy.abs(numpy.divide(kz,k))!=0.], bins=freq[1:30][:], weights=bias.real[numpy.abs(numpy.divide(kz,k))!=0.])
    dval, bins = numpy.histogram(k[numpy.abs(numpy.divide(kz,k))!=0.], bins=freq[1:30][:], weights=dbias.real[numpy.abs(numpy.divide(kz,k))!=0.])
    tval, bins = numpy.histogram(k[numpy.abs(numpy.divide(kz,k))!=0.], bins=freq[1:30][:], weights=theory.real[numpy.abs(numpy.divide(kz,k))!=0.])
    weight, bins = numpy.histogram(k[numpy.abs(numpy.divide(kz,k))!=0.], bins=freq[1:30][:])
    meanr.append(numpy.divide(rval, weight))
    meanb.append(numpy.divide(bval, weight))
    meant.append(numpy.divide(tval, weight))
    meand.append(numpy.divide(dval, weight))

fig, ax = plt.subplots(nrows=2, dpi=200, figsize=(6,6), sharex=True)
ax[0].semilogx((bins[1:] + bins[:-1])[:23]/2, numpy.mean(numpy.array(meanr),axis=0)[:23], lw=2, label='Simulation')
ax[0].semilogx((bins[1:] + bins[:-1])[:23]/2, numpy.mean(numpy.array(meant),axis=0)[:23], lw=2, label='Theory')
ax[1].semilogx((bins[1:] + bins[:-1])[:23]/2, numpy.mean(numpy.array(meanb),axis=0)[:23], lw=2, label=r'$P^{\mathrm{fid}}_{ge}=P^{\mathrm{true}}_{ge}$')
ax[1].semilogx((bins[1:] + bins[:-1])[:23]/2, numpy.mean(numpy.array(meand),axis=0)[:23], lw=2, label=r'${P^{\mathrm{fid}}_{ge}=P^{\mathrm{true}}_{ge}e^{-\frac{k^2}{k_{0}^2}}}$')
ax[0].set_ylabel(r'${r}$', fontsize=15);
ax[1].set_xlabel(r'${k \ [\mathrm{Mpc}^{-1}]}$', fontsize=15);
ax[1].set_ylabel(r'${b_v}$', fontsize=15);
ax[0].tick_params(axis='x', labelsize=10)
ax[1].tick_params(axis='x', labelsize=10)
ax[0].tick_params(axis='y', labelsize=10)
ax[1].tick_params(axis='y', labelsize=10)
ax[0].grid(ls='--', alpha=0.5, which='both')
ax[1].grid(ls='--', alpha=0.5, which='both')
ax[1].axvline(0.012, ls='--', color='grey')
ax[0].legend(frameon=False, fontsize=15)
ax[1].legend(frameon=False, fontsize=15, loc=2)
ax[0].set_ylim(0)
ax[1].set_ylim(0)
plt.tight_layout()
utensils.save_and_upload_plot(filename='figure3.pdf')


