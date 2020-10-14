import numpy
import joblib
import pathlib
import utensils
import kineticsz
from matplotlib import pyplot as plt
from nbodykit.lab import ArrayMesh, FFTPower, ProjectedFFTPower

plt.rcParams['text.usetex'] = True

class Powerspectrum:
    def __init__(self, pk, simulation=None, filename=None):
        self.k = None
        self.powerspectrum = None
 
        if isinstance(pk, (FFTPower, ProjectedFFTPower)):
            self.k = pk.power['k'].real
            self.powerspectrum = pk.power['power'].real
        elif isinstance(pk, kineticsz.Density):
            assert simulation is not None
            pk = FFTPower(ArrayMesh(pk.real/(simulation.boxsize/simulation.voxels)**3, BoxSize=simulation.boxsize), mode='1d')
            self.k = pk.power['k'].real
            self.powerspectrum = pk.power['power'].real
        
        elif isinstance(pk, numpy.ndarray):
            assert simulation is not None
            pk = FFTPower(ArrayMesh(pk/(simulation.boxsize/simulation.voxels)**3, BoxSize=simulation.boxsize), mode='1d')
            self.k = pk.power['k'].real
            self.powerspectrum = pk.power['power'].real


        if filename is not None:
            if simulation:
                self.save(filename=simulation.dirname.joinpath(filename))
            else:
                self.save(filename=filename)
    
    def plot(self, dpi=None):
        fig, ax = plt.subplots(dpi=dpi)
        ax.loglog(self.k, self.powerspectrum)
        ax.grid(ls='--', alpha=0.6)
        ax.set_xlabel('$k$')
        ax.set_ylabel('$P(k)$')
    
    def upload(self, filename=None, hashit=False):
        if filename is None:
            filename = 'pk.pdf'
            hashit = True
        fig, ax = plt.subplots(dpi=500)
        ax.loglog(self.k, self.powerspectrum)
        ax.grid(ls='--', alpha=0.6)
        ax.set_xlabel('$k$')
        ax.set_ylabel('$P(k)$')
        utensils.save_and_upload_plot(filename, hashit=hashit) 

    def save(self, filename):
        joblib.dump(self, pathlib.Path(filename).with_suffix('.powerspectrum'))
    
    @classmethod
    def load(cls, filename):
        return joblib.load(pathlib.Path(filename).with_suffix('.powerspectrum'))
