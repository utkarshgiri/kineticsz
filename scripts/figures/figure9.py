import fire
import h5py
import numpy
import scipy
import emcee
import pathlib
import utensils
import kineticsz
from scipy import stats
from matplotlib import pyplot as plt
plt.style.use(['science', 'ieee'])


def main(bins=12):
    binnums = bins
    velocity_samples = sorted(list(pathlib.Path('/gpfs/ugiri/mcmc_chains/').glob('velocity_samples*.h5')))
    density_samples = sorted(list(pathlib.Path('/gpfs/ugiri/mcmc_chains').glob('density_samples*.h5')))
    combined = h5py.File('/home/ugiri/kineticsz_personal/data/combined_velocity_quijote_58.h5', 'r')['mcmc']['chain'][-200:,:,-1].ravel()*numpy.sqrt(len(velocity_samples))

    dmedian = []; vmedian = []; percentile = []
    for vname, dname in zip(velocity_samples, density_samples):
        vm = h5py.File(vname, 'r')['mcmc']['chain'][-200:,:,-1]
        dm = h5py.File(dname, 'r')['mcmc']['chain'][-200:,:,-1]
        vmedian.append(numpy.median(vm))
        dmedian.append(numpy.median(dm))

    fig, ax = plt.subplots(ncols=1, dpi=500, figsize=(8,3))

    bins = numpy.linspace(-35,35,30)
    mu, sigma = scipy.stats.norm.fit(combined)
    a, b = kineticsz.utils.bootstrap(vmedian, numpy.std)
    best_fit_line = scipy.stats.norm.pdf(bins, 0, 11.54)
    ax.plot(bins, best_fit_line, ls='--', lw=3, label='Fisher forecast')
    _, bins, _ = ax.hist(vmedian, bins=numpy.linspace(-35,35,binnums), density=True, label='MCMC estimates', histtype='step', lw=2);
    ax.set_xlabel(r'$f_{NL}$')
    ax.grid(ls='-.')
    ax.legend()
    utensils.save_and_upload_plot(filename='figures/figure9.pdf')

if '__main__' == __name__:
    fire.Fire(main)
