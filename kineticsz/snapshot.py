import h5py
import numpy
import scipy
import logging
import kineticsz
from pathlib import Path
from kineticsz.utils import Powerspectrum
from nbodykit.source.mesh.catalog import CompensateTSC
from nbodykit.lab import ArrayCatalog, FFTPower, ProjectedFFTPower

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Snapshot:

    def __init__(self, snapshot, voxels=1024, in_units='mpc', out_units='mpc', 
                    dtype=numpy.float32, lazy=False, load=True, mute=True):
        
        snapshot = list(snapshot) if isinstance(snapshot, list) else [snapshot]    
        self.snapshot0 = snapshot[0]
        self.snapshot_list = snapshot
        self.handle = h5py.File(self.snapshot0, 'r')
        self.attrs = dict(list(self.handle['Header'].attrs.items()))
        
        self.dtype = dtype
        self.voxels = voxels
        self.in_units = in_units
        self.out_units = out_units

        if in_units == 'kpch' and out_units == 'mpc':
            self.conversion_factor = numpy.float32(1./(1000*self.hubble))
        if in_units == 'mpc' and out_units == 'mpc':
            self.conversion_factor = numpy.float32(1.)
        if in_units == 'kpch' and out_units == 'mpch':
            self.conversion_factor = numpy.float32(1./(1000))
        if in_units == 'kpc' and out_units == 'mpc':
            self.conversion_factor = numpy.float32(1./(1000))
        
        if lazy:
            logger.warning('Performing lazy evaluation. position and velocity will be filled with zeros')
            self.position = numpy.zeros((self.numparticle, 3), dtype=numpy.float32)
            self.peculiar_vz = numpy.zeros(self.numparticle, dtype=numpy.float32)
        else: 
            self.position = self.get_position()
            self.peculiar_vz = self.get_peculiar_vz()

        self.nbodykit_factor = (self.boxsize/self.voxels)**3
        self.nyquist = numpy.pi*self.voxels/self.boxsize
        self.fundamental_mode = 2*numpy.pi/self.boxsize

        if not mute: logger.info('Snapshot settings: \n {}'.format({x:y for (x,y) in self.__dict__.items() if isinstance(y, (float, int, bool, str))}))


    def get_handle(self):
        return self.handle
    
    def get_attrs(self)-> dict:
        return self.attrs
     
    def _combined_handle(self, particle, feature, axis=None):

        if axis is None:
            axis = [0,1,2]
        else:
            assert axis in [0,1,2]
            axis = [axis]

        assert feature in ['Coordinates', 'Velocities']

        array = numpy.array([]).reshape(-1, len(axis))

        for filename in self.snapshot_list:
            handle = h5py.File(filename, 'r')
            array = numpy.concatenate((array, handle[particle][feature][:, axis]), axis=0)

        return numpy.array(array, dtype=self.dtype)
     
     
    def get_position(self, particle='PartType1'):
        try: 
            if not numpy.all(self.position[:5]):
                self.position = self._combined_handle(particle=particle, feature='Coordinates')*self.conversion_factor
            return self.position
        except:
            return self._combined_handle(particle=particle, feature='Coordinates')*self.conversion_factor


    def x(self, particle='PartType1'):

        return self._combined_handle(particle=particle, feature='Coordinates', axis=0)*self.conversion_factor

    def y(self, particle='PartType1'):

        return self._combined_handle(particle=particle, feature='Coordinates', axis=1)*self.conversion_factor

    def z(self, particle='PartType1'):

        return self._combined_handle(particle=particle, feature='Coordinates', axis=2)*self.conversion_factor
    

    def vx(self, particle='PartType1'):

        return numpy.squeeze(self._combined_handle(particle=particle, feature='Velocities', axis=0))

    def vy(self, particle='PartType1'):

        return numpy.squeeze(self._combined_handle(particle=particle, feature='Velocities', axis=1))

    def vz(self, particle='PartType1'):

        return numpy.squeeze(self._combined_handle(particle=particle, feature='Velocities', axis=2))
   
    def get_peculiar_vx(self, particle='PartType1'):
        
        return numpy.array(self.vx(particle=particle)*self.velocity_factor, dtype=self.dtype)

    def get_peculiar_vy(self, particle='PartType1'):
        
        return numpy.array(self.vy(particle=particle)*self.velocity_factor, dtype=self.dtype)
        
    def get_peculiar_vz(self, particle='PartType1'):
        try: 
            if not numpy.all(self.peculiar_vz[:5]):
                self.peculiar_vz = numpy.array(self.vz(particle=particle)*self.velocity_factor, dtype=self.dtype)
            return self.peculiar_vz
        except:
            return numpy.array(self.vz(particle=particle)*self.velocity_factor, dtype=self.dtype)


    @property
    def name(self):
        '''
        Returns the name of the snapshot file
        '''
        return self.snapshot
    
    @property
    def get_snapshots(self):
        '''
        Returns the name of the snapshot file
        '''
        return self.snapshot_list


    @property
    def dirname(self):
        '''
        Returns the name of the snapshot file
        '''
        return Path(self.snapshot0).parent


    @property
    def tag(self):
        '''
        This function gives the name of directory
        storing the snapshot. This can be used as
        a unique tag to save processed results of
        from a given simulation.

        Note: This tag is unique at simulation 
        level and not at snapshot level
        '''
        name = str(str(self.name).split('/')[-2]) + 'z%s'%round(self.redshift, 2)
        return name

    @property
    def velocity_factor(self):
        '''
        This function returns the square root of
        scale factor a which is  multiplied 
        to the velocity to get comoving velocity
        '''

        return numpy.float32(numpy.sqrt(1./(1+self.redshift)))

    @property
    def boxsize(self):
        boxsize = round(float(self.attrs['BoxSize'])*self.conversion_factor,4)
        return boxsize

    @property
    def redshift(self):
        return float(self.attrs['Redshift'])

    @property
    def scale_factor(self):
        return float(self.attrs['Time'])

    @property
    def numparticle(self):
        return self.attrs['NumPart_Total'].sum()

    @property
    def hubble(self):
        return float(self.attrs['HubbleParam'])

    @property
    def DM_mass(self):
        return self.attrs['MassTable'][1]*1e10


    def matter_powerspectrum(self, load=False, voxels:int=None, interlacing=False, resampler='TSC'):
        """Compute matter powerspectrum for the snapshot

        Args:
            load (bool, optional): If true, try loading from disk. Defaults to False.
            voxels (int, optional): If not None, use this as the voxel size. Defaults to None.
            interlacing (bool, optional): If true, perform interlacing correction. Defaults to False.
            resampler (str, optional): The painting scheme to be used. Defaults to 'TSC'.

        Returns:
            Powerspectrum: Powerspectrum object containing matter powerspectrum
        """
        filename = Path(self.dirname).joinpath('matter.powerspectrum')
        
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            powerspectrum = Powerspectrum.load(filename)
            return powerspectrum
        
        position = self.position
        voxels = voxels if voxels is not None else self.voxels
        catalog = ArrayCatalog({'position': position})
        if interlacing:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', resampler=resampler, compensated=False, interlaced=True)
            mesh = mesh.apply(CompensateTSC, kind='circular', mode='complex')
        else:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', compensated=True)
        powerspectrum = Powerspectrum(FFTPower(mesh, mode='1d'))
        powerspectrum.save(filename)
        return powerspectrum

    def ksz_powerspectrum(self, load=False, voxels:int=None, interlacing=False, resampler='TSC'):
        """Compute kSZ powerspectrum by projecting snapshot momentum to x-y surface

        Args:
            load (bool, optional): If true, try loading from disk. Defaults to False.
            voxels (int, optional): If not None, use this as the voxel size. Defaults to None.
            interlacing (bool, optional): If true, perform interlacing correction. Defaults to False.
            resampler (str, optional): The painting scheme to be used. Defaults to 'TSC'.

        Returns:
            Powerspectrum: kSZ powerspectrum
        """
        filename = Path(self.dirname).joinpath('ksz.powerspectrum')
        
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            powerspectrum = Powerspectrum.load(filename)
            return powerspectrum
        
        position = self.position
        voxels = voxels if voxels is not None else self.voxels
        catalog = ArrayCatalog({'position': position, 'velocity': self.peculiar_vz})
        if interlacing:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', value='velocity', resampler=resampler, compensated=False, interlaced=True)
            mesh = mesh.apply(CompensateTSC, kind='circular', mode='complex')
        else:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', value='velocity', compensated=True)

        powerspectrum = Powerspectrum(ProjectedFFTPower(mesh))
        powerspectrum.powerspectrum *= self.boxsize**2
        powerspectrum.save(filename)
        return powerspectrum

    def density_field(self, load=False, mode='real', save=False):
        """Computes and returns the matter density field from snapshot

        Args:
            load (bool, optional): If true, try loading from disk. Defaults to False.
            mode (str, optional): Desired density format. Could be `real` or `complex`. Defaults to 'real'.
            save (bool, optional): If true, save density to disk. Defaults to False.

        Returns:
            kineticsz.Density: Returns matter density
        """
        filename = self.dirname.joinpath('matter.density')
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()}')
            density = kineticsz.Density.load(filename)
        else:
            position = self.position
            catalog = ArrayCatalog({'position': position})
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position')
            density = kineticsz.Density(density=numpy.array(mesh.paint(mode=mode)*self.nbodykit_factor), voxels=self.voxels, boxsize=self.boxsize, redshift=self.redshift) 
            if save: density.save(filename)
        
        return density

    def density_contrast(self, mode='real', load=False):
        """Computes and returns matter overdensity from snapshot data

        Args:
            mode (str, optional): The format in which the overdensity is desired. Defaults to 'real'.
            load (bool, optional): If true, try loading from disk. Defaults to False.

        Returns:
            [type]: [description]
        """
        filename = self.dirname.joinpath('matter-contrast.density')
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            mesh = numpy.load(filename)
        else:
            mesh = self.density_field(contrast=False)
            mesh /= numpy.mean(mesh, dtype=numpy.float64)
            mesh -= 1.
            numpy.save(filename, mesh)
        if mode == 'complex':
            return scipy.fft.rfftn(mesh)
        return mesh


    def weighted_field(self, weight: numpy.ndarray, mode='real'):
        """Computes and returns a weighted density field

        Args:
            weight (numpy.ndarray): Array with which to weight the particle position.
            mode (str, optional): `real` or `complex`. Defaults to 'real'.

        Returns:
            numpy.ndarray: Array containing the weighted field
        """

        position = self.position
        catalog = ArrayCatalog({'position': position, 'weight': weight})
        mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position', value='weight')
        mesh = mesh.paint(mode=mode)
        return numpy.array(mesh)*self.nbodykit_factor


    def radial_momentum_field(self, load=False, mode='real', save=False):
        """Computes and returns the radial momentum field

        Args:
            load (bool, optional): If true, try loading from disk. Defaults to False.
            mode (str, optional): `real` or `complex`. Defaults to 'real'.
            save (bool, optional): Save to disk if true. Defaults to False.

        Returns:
            kineticsz.Density: radial momentum density
        """
        filename = self.dirname.joinpath('momentum.density')
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            density = kineticsz.Density.load(filename)
        else:
            position = self.position
            velocity = self.peculiar_vz if self.peculiar_vz is not None else self.get_peculiar_vz()
            catalog = ArrayCatalog({'position': position, 'weight': velocity})
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position', value='weight', compensated=True)
            density = kineticsz.Density(numpy.array(mesh.paint(mode=mode)*self.nbodykit_factor), voxels=self.voxels, boxsize=self.boxsize, redshift=self.redshift)
            if save: density.save(filename)
        
        return density


    def matter_halo_powerspectrum(self,  minmass=0, maxmass=1e20, load=False, voxels=None, interlacing=False, resampler='TSC'):
        """Calculate matter-halo cross powerspectrum from the snapshot and corresponding rockstar catalogue

        Args:
            minmass (float, optional): Minimum halo mass cutoff. Defaults to 0.
            maxmass (float, optional): Maximum halo mass cutoff. Defaults to 1e20.
            load (bool, optional): If true, try loading from disk. Defaults to False.
            voxels (int, optional): If not None, use this for gridding. Defaults to None.
            interlacing (bool, optional): If true, perform interlacing. Defaults to False.
            resampler (str, optional): Painting scheme to be used. Defaults to 'TSC'.

        Returns:
            kineticsz.utils.Powerspectrum: [description]
        """

        filename = Path(self.dirname).joinpath('matter-halo.powerspectrum')
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            cross_powerspectrum = Powerspectrum.load(filename)
            return cross_powerspectrum
 
        rockstar = kineticsz.Rockstar(rockstar=self.dirname.joinpath('halos_0.0.ascii').as_posix(), out_units=self.out_units, minmass=minmass, maxmass=maxmass)
        position = self.position
        voxels = voxels if voxels is not None else self.voxels
        snapcatalog = ArrayCatalog({'position': position})
        if interlacing:
            snapmesh = snapcatalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', resampler=resampler, compensated=False, interlaced=True)
            snapmesh = snapmesh.apply(CompensateTSC, kind='circular', mode='complex')
        else:
            snapmesh = snapcatalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', compensated=True)
        catalog = ArrayCatalog({'position': rockstar.position})
        if interlacing:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', resampler=resampler, compensated=False, interlaced=True)
            mesh = mesh.apply(CompensateTSC, kind='circular', mode='complex')
        else:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', compensated=True)
        cross_powerspectrum = Powerspectrum(FFTPower(mesh, second=snapmesh, mode='1d'))
        cross_powerspectrum.save(filename)
        return cross_powerspectrum



    def momentum_powerspectrum(self, load=False, voxels=None, interlacing=False, resampler='TSC'):
        """Calculates and returns radial momentum powerspectrum

        Args:
            load (bool, optional): If true, try loading from disk. Defaults to False.
            voxels ([type], optional): If not None, use this for gridding. Defaults to None.
            interlacing (bool, optional): If true, perform interlacing. Defaults to False.
            resampler (str, optional): Painting scheme to use. Defaults to 'TSC'.

        Returns:
            : [description]
        """
        
        filename = Path(self.dirname).joinpath('momentum.powerspectrum')
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            momentum_powerspectrum = Powerspectrum.load(filename)
            return momentum_powerspectrum
        
        position = self.position
        voxels = voxels if voxels is not None else self.voxels
        velocity = self.peculiar_vz if self.peculiar_vz is not None else self.get_peculiar_vz()
 
        catalog = ArrayCatalog({'position': position, 'vz': velocity})
        if interlacing:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', value='vz', resampler=resampler, compensated=False, interlaced=True)
            mesh = mesh.apply(CompensateTSC, kind='circular', mode='complex')
        else:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', value='vz', compensated=True)

        powerspectrum = Powerspectrum(FFTPower(mesh, mode='1d'))
        powerspectrum.save(filename)
        return powerspectrum



