import numpy
import scipy
import joblib
import logging
import pathlib
import utensils
import matplotlib
from matplotlib import pyplot as plt
plt.style.use(['science'])

dirnames = sorted(list(pathlib.Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('*/snapdir_001/reconstructed-velocity.powerspectrum')), key=lambda x: x.stat().st_mtime)

vpk, qpk, npk, rpk, upk, eta = ([] for _ in range(6))
for name in dirnames[:]:
    try:
        name = name.parent
        num = str(name).split('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/')[-1].split('/snapdir_001')[0]
        h1 = joblib.load(name.joinpath('reconstructed-velocity.powerspectrum'))
        h2 = joblib.load(name.joinpath('noise.powerspectrum'))
        h3 = joblib.load(name.joinpath('residual-noise.powerspectrum'))
        h5 = joblib.load(name.joinpath('momentum.powerspectrum'))
        vpk.append(h1.powerspectrum)
        npk.append(h2.powerspectrum)
        rpk.append(h3.powerspectrum)
        qpk.append(h5.powerspectrum)
    except:
        print('{} failed'.format(name))
        continue
print(f'{len(vpk)} files found!')

fig, ax = plt.subplots(dpi=500, figsize=(6, 4))
end = 256; ls=(0, (5, 1)); lw=2
k = joblib.load(dirnames[0].parent.joinpath('momentum.powerspectrum')).k[:end]
c = scipy.constants.c/1e3

ax.loglog(k[1:end], numpy.array(vpk).mean(0)[1:end]/c**2, lw=lw, ls=ls, label='$\mathrm{Reconstruction} \ P_{\hat{v}_r}$')
ax.loglog(k[1:end], numpy.array(qpk).mean(0)[1:end]/c**2, lw=lw, ls='dashdot', label='$\mathrm{Momentum} \ P_{q_r}$')
ax.loglog(k[1:end], numpy.array(rpk).mean(0)[1:end]/c**2, lw=lw, ls=(0, (3, 1, 1, 1)), label='$\mathrm{Noise} \ P_{\eta}$')
ax.loglog(k[1:end], numpy.array(npk).mean(0)[1:end]/c**2, lw=lw, ls=(0, (3, 1, 1, 1, 1, 1)), label='$N^{\mathrm{(0)}}\mathrm{-bias}$')
ax.legend(frameon=False, fontsize=12)
ax.grid(which='both',ls='-.', alpha=0.5)
ax.set_ylabel(r'${P(k) \ [\mathrm{Mpc}^{3}]}$')
ax.set_xlabel(r'${k \ [\mathrm{Mpc}^{-1}]}$');
plt.tight_layout()
utensils.save_and_upload_plot(filename='figures/figure4.pdf', dpi=500)
