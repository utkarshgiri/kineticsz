import fire
import numpy
import utensils
import kineticsz
from  classylss import binding
from scipy.integrate import simps
from matplotlib import pyplot as plt
plt.style.use('science')

def main(plot=True):

    simulation = kineticsz.Simulation('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/99/snapdir_001/', in_units='kpch', lazy=True)
    kint = numpy.linspace(simulation.snapshot.fundamental_mode, simulation.snapshot.nyquist, simulation.snapshot.voxels)
    epsilon = 1e-6
    muint = numpy.linspace(-1+epsilon,1-epsilon, simulation.voxels, endpoint=False)
    cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/ics/configuration/class_quijote.json', units='mpc')
    pmm_lin = cmb.interpolate_linear_powerspectrum(redshift=simulation.redshift)
    pmm_nl = cmb.interpolate_nonlinear_powerspectrum(redshift=simulation.redshift)
    classy = binding.ClassEngine(cmb.parameters)
    background = binding.Background(classy)
    f = background.scale_independent_growth_rate(cmb.redshift)
    h = background.hubble_function(cmb.redshift)*cmb.c
    a = 1./(1. + simulation.redshift)
    factor = f * a * h
    kstar = cmb.Kksz(simulation.redshift)
    chistar = cmb.Chistar(simulation.redshift)
    Ik = numpy.zeros_like(kint)
    for i,k in enumerate(kint):

        def F(mu, kp):
            return (kp**2/((4*numpy.pi**2))*pmm_nl(numpy.sqrt((k**2+kp**2-2*kp*k*mu)))*pmm_nl(kp)*((k*(k-2*kp*mu)*(1-mu**2))/(kp**2*(k**2+kp**2-2*kp*k*mu))))
        
        z = F(muint.reshape(-1,1), kint.reshape(1,-1))
        Ik[i] = factor**2*simps([simps(zmu, muint) for zmu in z], kint)
    print(Ik)
    numpy.savez('../data/numerical_SkSZ.npz', k=kint, powerspectrum=Ik)
    if plot:
        l = chistar*kint
        _, ax = plt.subplots()
        ax.semilogy(l[l>500], kstar**2*kint[l>500]**2*simulation.boxsize*numpy.array(Ik)[l>500]/2/(2*numpy.pi),  lw=2, label='$C_l^{SkSZ}$', ls='--')
        ax.set_xlabel('k')
        ax.set_ylabel('Pk')
        utensils.save_and_upload_plot(filename='../data/SkSZ.pdf')

if '__main__' == __name__:
    fire.Fire(main)
