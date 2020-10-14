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
class Simulation:

    dirname : Union[PosixPath, str] = None
    snapshot: Union[PosixPath, str] = None
    rockstar: Union[PosixPath, str] = None
    lazy: bool = False
    dtype: numpy.dtype = numpy.float32
    voxels: int = 1024
    boxsize: float = None
    redshift: float = 2
    in_units: str = 'kpch'
    out_units: str = 'mpc'
    minmass: float = 1e0
    maxmass: float = 1e20
    number_density: float = None
    class_parameters : Union[dict, PosixPath, str] = None
    
    def __post_init__(self):

        if self.dirname is None:
            assert self.snapshot is not None, "Both 'dirname' and 'snapshot' is None"
            assert self.rockstar is not None, "Both 'dirname' and 'rockstar' is None"
            assert Path(str(self.snapshot)).parent == Path(str(self.rockstar)).parent
        else:
            try:
                self.snapshot, self.rockstar = kineticsz.utils.find_snapshot_and_catalog(directory=self.dirname, redshift=self.redshift)
            except:
                logger.exception('Exception while looking for snapshot and catalogue in {}'.format(self.dirname))

        self.dirname = Path(self.rockstar).parent
        self.snapshot = kineticsz.Snapshot(snapshot=self.snapshot, voxels=self.voxels, in_units=self.in_units, 
                                            out_units=self.out_units, dtype=self.dtype, lazy=self.lazy)
        self.rockstar = kineticsz.Rockstar(rockstar=self.rockstar, voxels=self.voxels, in_units='mpch', 
                                            out_units=self.out_units, dtype=self.dtype, minmass=self.minmass, maxmass=self.maxmass)
        
        if self.boxsize is None:
            self.boxsize = self.snapshot.boxsize
        assert numpy.isclose(self.boxsize, self.rockstar.boxsize), f'Snapshot boxsize is {self.boxsize}, Rockstar boxsize is {self.rockstar.boxsize}'
        self.number_density = self.snapshot.numparticle/self.boxsize**3


    @property
    def hubble(self):
        return self.rockstar.hubble

    def is_compatible_with(self, tmap):
        
        if (isinstance(tmap, kineticsz.Map) & numpy.isclose(self.snapshot.boxsize, tmap.boxsize) &
                numpy.isclose(self.redshift, tmap.redshift, atol=0.01)):
            return True
        else:
            return False

    def save(self, filename):
        joblib.dump(self, Path(filename).with_suffix('.simulation'))

    @classmethod
    def load(self, filename):
        joblib.dump(self, Path(filename).with_suffix('.simulation'))
