import numpy
import scipy
import joblib
import logging
import kineticsz
from pathlib import Path
from nbodykit.lab import *
from dataclasses import dataclass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class Density:
    #The density should be properly normalized. For nbodykit based painting, this means multiplying by an nbodykit_factor
    density: numpy.ndarray
    simulation: kineticsz.Simulation = None
    voxels: int = None
    boxsize: float = None
    redshift: float = None
    
    def __post_init__(self):
        
        if self.simulation is not None:
            self.voxels = self.simulation.voxels if self.voxels is None else self.voxels
            self.boxsize = self.simulation.boxsize if self.boxsize is None else self.boxsize
            self.redshift = self.simulation.redshift if self.redshift is None else self.redshift

        self.forward = self.boxsize/self.voxels
        self.backward = 1./self.forward 
        if self.density.shape == (self.voxels, self.voxels, int(self.voxels/2 + 1)):
            self.density = self.backward**3*scipy.fft.irfftn(self.density, workers=20)

    def __array__(self):
        return self.density
    
    def __getitem__(self, index):
        return self.density.__getitem__(index)

    def __setitem__(self, index, value):
        return self.density.__setitem__(index, value)

   
    @property
    def complex(self):
        """Returns the density field in fourier space

        Returns:
            numpy.ndarray: Array of map values in fourier space
        """
        if self.density.shape == (self.voxels, self.voxels, self.voxels):
            #return self.forward**3*scipy.fft.rfftn(self.density, workers=20).astype(numpy.complex64)
            return scipy.fft.rfftn(self.density, workers=20).astype(numpy.complex64)
        else:
            return self.density

    @property
    def real(self):
        """Returns the density field in real space

        Returns:
            numpy.ndarray: Array of density field values in real space
        """
        if self.density.shape == (self.voxels, self.voxels, int(self.voxels/2 + 1)):
            return self.backward**3*scipy.fft.irfftn(self.density, workers=20)
        else:
            return self.density


    def __mul__(self, second):
        """Implements multiplication of number by density field

        Args:
            second ([type]): object which gets multiplied

        Returns:
            Density: Result of multiplication
        """              
        if numpy.isscalar(second):
            return Density(second*self.density.real, self.voxels, self.boxsize, self.redshift)

        else:
            return NotImplemented


    def __rmul__(self, second):
        """Implements multiplication of number by density field

        Args:
            second ([type]): object which gets multiplied

        Returns:
            Density: Result of multiplication
        """        
        if numpy.isscalar(second):
            return Density(second*self.density.real, self.voxels, self.boxsize, self.redshift)

        else:
            return NotImplemented


    def __matmul__(self, second):
        if isinstance(second, Density):
            return Density(self.complex * second.complex, self.voxels, self.boxsize, self.redshift)
        elif isinstance(second, numpy.ndarray):
            assert self.complex.shape == second.shape
            return Density(self.complex * second, self.voxels, self.boxsize, self.redshift)



    def __truediv__(self, second):
        """Implements division of density field by a number

        Args:
            second ([type]): number by which to divide

        Returns:
            Density: Result of division
        """

        if numpy.isscalar(second):
            return Density(self.density/second, self.voxels, self.boxsize, self.redshift)
        else:
            NotImplemented

    def __rtruediv__(self, second):
        """Implements division of number by density field

        Args:
            second ([type]): object which gets divided

        Returns:
            Density: Result of division
        """

        if numpy.isscalar(second):
            return Density(second/self.density, self.voxels, self.boxsize, self.redshift)


    def filter(self, weight: numpy.ndarray):
        """This function applies filter to the density in fourier space

        Args:
            weight (numpy.ndarray): Weight filter to be used

        Returns:
            Density: The filtered density
        """
        
        assert weight.shape == self.complex.shape
        return Density(self.complex * weight, self.voxels, self.boxsize, self.redshift)

    def save(self, filename):
        """A helper method to save the instance

        Args:
            filename ([type]): Name of the file 
        """
        self.simulation = None
        joblib.dump(self, Path(filename).with_suffix('.density'))
    
    @classmethod
    def load(cls, filename):
        """A class method to load density object from disk

        Args:
            filename ([type]): Name of the file to load from disk

        Returns:
            Density: The loaded density object
        """
        return joblib.load(Path(filename).with_suffix('.density'))

    def sum(self, axis: int=2):
        """A helper method to sum up density field along an axis

        Args:
            axis (int, optional): axis number. Defaults to 2.

        Returns:
            Map: 2D map of projected density
        """
        return kineticsz.Map(self.density.real.sum(axis), self.voxels, self.boxsize, self.redshift)


    def powerspectrum(self, filename):
        """Computes, saves and returns the powerspectrum of the density field

        Args:
            filename ([type]): Name of file where to save the powerspectrum

        Returns:
            Powerspectrum: [description]
        """
        pk = kineticsz.utils.Powerspectrum(FFTPower(ArrayMesh(self.real/(self.boxsize/self.voxels)**3, BoxSize=self.boxsize), mode='1d'), simulation=self.simulation, filename=filename)
        return pk
