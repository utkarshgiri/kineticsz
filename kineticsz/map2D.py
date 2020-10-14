import numpy
import scipy
import joblib
import logging
import kineticsz
from typing import Union
from dataclasses import dataclass
from pathlib import Path, PosixPath
from kineticsz.utils import Powerspectrum

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class Map:
    map : numpy.ndarray
    pixels: int
    boxsize : float
    redshift : float
    
    def __post_init__(self):
        self.forward = self.boxsize/self.voxels
        self.backward = 1./self.forward 
        if self.map.shape == (self.pixels, int(self.pixels/2 + 1)):
            self.map = self.backward**2*scipy.fft.irfft2(self.map, workers=20)


    def __array__(self):
        return self.map
    
    def __getitem__(self, index):
        return self.map.__getitem__(index)

    def __setitem__(self, index, value):
        return self.map.__setitem__(index, value)


    @property
    def complex(self):
        """Returns the map in fourier space

        Returns:
            numpy.ndarray: Array of map values in fourier space
        """
        if self.map.shape == (self.pixels, self.pixels):
            return self.forward**2*scipy.fft.rfft2(self.map)
        else:
            return self.map
    
    @property
    def real(self):
        """Returns the map in configuration space

        Returns:
            numpy.ndarray: Array of map values in real space
        """
        if self.map.shape == (self.pixels, int(self.pixels/2 + 1)):
            return self.backward**2*scipy.fft.irfft2(self.map, workers=20)
        else:
            return self.map

    def __add__(self, second):
        """Adds a second map to the primary map

        Args:
            second ([Map]): The map to be added

        Returns:
            [Map]: The resultant Map
        """
        assert self.compatible(second), 'The maps are incompatible. Cannot be added'
        return Map(self.real + second.real, self.pixels, self.boxsize, self.redshift)
    
    def __mul__(self, second):
        
        if numpy.isscalar(second):
            return Map(second*self.real[:], self.pixels, self.boxsize, self.redshift)

        elif isinstance(second, kineticsz.Density):
            #assert second.real.dtype == numpy.floating
            assert second.real.ndim == 3
            return kineticsz.Density(second.real * self.real[:,:,None], voxels=second.voxels, boxsize=second.boxsize, redshift=second.redshift)

        elif isinstance(second, numpy.ndarray):
            assert second.ndim == 3
            assert second.dtype == numpy.floating
            return second * self.real[:,:,None]
       
        else:
            return NotImplemented


    def __rmul__(self, second):
        
        if numpy.isscalar(second):
            return Map(second*self.real[:], self.pixels, self.boxsize, self.redshift)

        elif isinstance(second, kineticsz.Density):
            assert second.real.dtype == numpy.floating
            assert second.real.ndim == 3
            return kineticsz.Density(second.real * self.real[:,:,None], voxels=second.voxels, boxsize=second.boxsize, redshift=second.redshift)

        elif isinstance(second, numpy.ndarray):
            assert second.ndim == 3
            assert second.dtype == numpy.floating
            return second * self.real[:,:,None]
       
        else:
            return NotImplemented


    def compatible(self, second):
        """A function to test whether a second map is compatible with self

        Args:
            second ([type]): A map object

        Returns:
            bool: True if compatible else False
        """

        condition = (isinstance(second, Map) & second.pixels == self.pixels & (second.boxsize == self.boxsize)
                    & (second.redshift == self.redshift) & (second.real.shape == self.real.shape))
        if condition:
            return True
        else:
            return False

    def filter(self, weight: Union[Powerspectrum, numpy.ndarray]):
        """This function applies filter to the 2D map in fourier space

        Args:
            weight (Union[Powerspectrum, numpy.ndarray]): Weight filter to be used

        Returns:
            Map: The filtered map
        """
        if isinstance(weight, Powerspectrum):
            weight = numpy.divide(1., numpy.interp(kineticsz.utils.k2D(self.boxsize, self.pixels), weight.k, weight.powerspectrum))
        
        assert weight.shape == self.complex.shape

        return Map(self.complex * weight, self.pixels, self.boxsize, self.redshift)

    def save(self, filename):
        """A helper method to save the instance

        Args:
            filename ([type]): Name of the file s
        """
        joblib.dump(self, Path(filename).with_suffix('.map'))
    
    @classmethod
    def load(cls, filename):
        """A class method to load map2D object from disk

        Args:
            filename ([type]): Name of the file to load from disk

        Returns:
            Map: The loaded map2D object
        """
        return joblib.load(Path(filename).with_suffix('.map'))



