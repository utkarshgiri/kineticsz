import numpy
import pandas
import logging
import linecache
import kineticsz
from pathlib import Path
from kineticsz import Density
from kineticsz.utils import Powerspectrum
from nbodykit.lab import ArrayCatalog, FFTPower
from nbodykit.source.mesh.catalog import CompensateTSC

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Rockstar:

    '''Class to read and analyze rockstar halo catalogues'''
    
    def __init__(self, rockstar: str, voxels=1024,  in_units: str='mpch', out_units :str='mpc',
                    minmass: float=1e10, maxmass: float=1e20, dtype=numpy.float32, min_particle=None, mute=True):
        
        self.units = out_units
        self._dtype = dtype
        self.minmass = minmass
        self.maxmass = maxmass
        self.voxels = voxels
        self.rockstar = rockstar
        self.frame = self.read_csv(rockstar, dtype)
        self.whole_frame = self.frame.copy()
        self.min_particle = self.frame['num_p'].min()

        if min_particle:
            assert self.min_particle == min_particle, f'Minimum number of particle per halo is not {min_particle} but {self.min_particle}'

        self.frame = self.frame[(self.frame['mvir'] > self.minmass) & (self.frame['mvir'] < self.maxmass)]
        if in_units == 'mpch' and out_units == 'mpc':
            self.frame[['x','y','z']] /= self.hubble

        self.position = self.get_position()
        self.nbodykit_factor = (self.boxsize/self.voxels)**3
        if not mute: logger.info('Rockstar settings: \n {}'.format({x:y for (x,y) in self.__dict__.items() if isinstance(y, (float, int, bool, str))}))


    def read_csv(self, rockstar, dtype)-> pandas.DataFrame:
        frame = pandas.read_csv(rockstar, sep=' ', header=0, skiprows=19, dtype=dtype)
        with open(rockstar, "r") as f:
            header = f.readline()
        frame.columns = header.split(" ")
        return frame


    def dataframe(self)-> pandas.DataFrame:
        '''Returns the dataframe of the halo catalogue'''
        return self.frame


    def get_position(self)-> numpy.ndarray:
        '''Returns the position of halos in an array shaped (N,3)'''
        coordinates = numpy.column_stack([self.frame.x, self.frame.y, self.frame.z])
        return coordinates

    def least_massive(self)-> numpy.int:
        '''Returns the mass of least massive halo. Mostly this should be `self.min_particle*DM_mass`'''
        return numpy.min(self.frame.mvir)

    @property
    def name(self):
        '''Returns the name of the halo file'''
        return self.rockstar

    @property
    def dirname(self):
        '''Returns the name of directory of the snapshot file'''
        return Path(self.name).parent
    
    
    @property
    def directory(self):
        '''Returns the name of directory of the snapshot file'''
        return Path(self.name).parent.as_posix()


    @property
    def boxsize(self)-> numpy.float:
        '''Returns the boxsize of the halo catalogue'''
        line = linecache.getline(self.rockstar, 7)
        size = float(line.split(':')[-1].split('Mpc')[0])
        if self.units == 'mpch':
            return round(size,4)
        elif self.units == 'mpc':
            return round(size/self.hubble,4)
        else:
            raise ValueError("The parameter units is neither mpc nor mpch")

    @property
    def scale_factor(self)-> float:
        '''Returns the scale factor of the snapshot'''
        line = linecache.getline(self.rockstar, 2)
        return round(float(line.strip().split('= ')[-1]),3) 
    
    @property
    def hubble(self):
        '''Returns the hubble parameter of the simulation'''
        line = linecache.getline(self.rockstar, 3)
        return numpy.float32(line.split('h =')[-1].strip())


    @property
    def num_halos(self)-> int:
        '''Returns the number of halos in the halo catalogue'''
        return len(self.frame)

    @property
    def redshift(self)-> float:
        '''Returns the redshift of the halo catalogue'''
        return (1./self.scale_factor) - 1


    def density_field(self, load=False, save=False, mode='real')-> Density:
        '''
        A function taht computes and returns the halo density field

        Args:
        load {Bool} : If True, load from disk if available
        save {bool} : If True, save to disk
        mode {str} : `real` or `complex`
        
        Returns:
        A Density object containing the halo density field
        '''

        filename = self.dirname.joinpath(f'halo')
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            density = Density.load(filename)
        else:
            catalog = ArrayCatalog({'position': self.position})
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position', compensated=True)
            density = Density(numpy.array(mesh.paint(mode=mode)*self.nbodykit_factor), voxels=self.voxels, boxsize=self.boxsize, redshift=self.redshift)
            if save: density.save(filename)

        return density


    def density_contrast(self, load=False, mode='real')-> Density:
        '''
        A function taht computes and returns the halo overdensity field

        Args:
        load {Bool} : If True, load from disk if available
        mode {str} : `real` or `complex`
        
        Returns:
        A Density object containing the halo overdensity field
        '''

        mesh = self.density_field(load=load, mode=mode)
        mesh.density /= numpy.mean(mesh.density, dtype=numpy.float64)
        mesh.density -= 1.
        return mesh


    def shot_noise(self, minmass=0, maxmass=1e20)-> numpy.float:
        '''Returns the shot noise of the catalogue'''

        if minmass:
            num_halos = len(self.whole_frame[(self.whole_frame['mvir'] > minmass) & (self.whole_frame['mvir'] < maxmass)])
        else:
            num_halos = self.num_halos

        return self.boxsize**3/num_halos


    def halo_powerspectrum(self, load=False, voxels=None, interlacing=False, resampler='TSC'):
        """Compute and return halo auto-powerspectrum

        Args:
            load (bool, optional): If True, try loading from disk. Defaults to False.
            voxels ([type], optional): If not None, use this for gridding. Defaults to None.
            interlacing (bool, optional): If True, perform interlacing. Defaults to False.
            resampler (str, optional): Painting scheme to be used. Defaults to 'TSC'.

        Returns:
            Powerspectrum: halo auto-powerspectrum
        """
        filename = Path(self.dirname).joinpath('halo')

        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            powerspectrum = Powerspectrum.load(filename)
            return powerspectrum
        
        frame = self.frame
        voxels = voxels if voxels is not None else self.voxels
        position = numpy.column_stack([frame.x, frame.y, frame.z])
        catalog = ArrayCatalog({'position': position})
        if interlacing:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', resampler=resampler, compensated=False, interlaced=True)
            mesh = mesh.apply(CompensateTSC, kind='circular', mode='complex')
        else:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=voxels, position='position', compensated=True)
        powerspectrum = Powerspectrum(FFTPower(mesh, mode='1d'))
        powerspectrum.save(filename) 
        return powerspectrum


    @property
    def corresponding_snapshot(self):
        '''
        This function searches for a snapshot file in the base directory
        where the halo catalogue resides and returns the one which has
        redshift very close to that of the halo vcatlogue.
        '''
        snapshot = []
        snapshots = Path(self.directory).glob('snap*.hdf5')
        for name in snapshots:
            s = kineticsz.Snapshot(name)
            if self.redshift - 0.01 < s.redshift < self.redshift + 0.01:
                snapshot.append(name)
        return snapshot


    def matter_halo_powerspectrum(self, snapshot_units='mpc', snapshot_out_units='mpc', minmass=0, maxmass=1e20, load=False, interlacing=False,resampler='TSC'):

        filename = Path(self.dirname).joinpath('matter-halo')
        if load and filename.exists():
            logger.info(f'Loading {filename.as_posix()} from disk ...')
            cross_powerspectrum = Powerspectrum.load(filename)
            return cross_powerspectrum
 
        snapshot = self.corresponding_snapshot

        if snapshot is None:
            return None
        
        snapshot = kineticsz.Snapshot(snapshot, in_units=snapshot_units, out_units=snapshot_out_units)
        snapcatalog = ArrayCatalog({'position': snapshot.position})
        snapmesh = snapcatalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position', compensated=True)

        #if minmass != 0 or maxmass != 1e20:
        #    frame = self.whole_frame[(self.whole_frame['mvir'] > minmass) & (self.whole_frame['mvir'] < maxmass)]
        #else:
        frame = self.frame

        position = numpy.column_stack([frame.x, frame.y, frame.z])
        catalog = ArrayCatalog({'position': position})
        if interlacing:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position', resampler=resampler, compensated=False, interlaced=True)
            mesh = mesh.apply(CompensateTSC, kind='circular', mode='complex')
        else:
            mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position', compensated=True)#, resampler=resampler, compensated=False, interlaced=True)
        cross_powerspectrum = Powerspectrum(FFTPower(mesh, second=snapmesh, mode='1d'))
        cross_powerspectrum.save(filename)
        return cross_powerspectrum


    def correlation(self, full: bool=False, bins: int=20):
        '''
        This function returns the cross correlation coefficient between 
        halo density and matter density. By default the correlation is binned 
        in k-bins but if full is True, then the 3D r-matrix is returned
        '''

        snapshot = self.corresponding_snapshot

        if snapshot is None:
            logger.info('Corresponding snapshot not found. Returning None')
            return None
        snapshot = kineticsz.Snapshot(snapshot)
        snapcatalog = ArrayCatalog({'position': snapshot.position})
        snapmesh = snapcatalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position')

        frame = self.frame
        position = numpy.column_stack([frame.x, frame.y, frame.z])
        catalog = ArrayCatalog({'position': position})
        halomesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position')

        halofield = numpy.array(halomesh.paint('complex')); snapfield = numpy.array(snapmesh.paint('complex'))

        r = halofield*snapfield.conjugate()/(numpy.sqrt(halofield*halofield.conjugate() * snapfield*snapfield.conjugate()))

        if full:
            return r

        frequency = 2*numpy.pi*numpy.fft.fftfreq(self.voxels, self.boxsize/self.voxels)
        norm = numpy.zeros_like(r)

        for i in range(self.voxels):
            for j in range(self.voxels):
                norm[i,j,:] = numpy.sqrt(frequency[i]**2 + frequency[j]**2 + frequency[:r.shape[2]]**2)
        bin_size = bins
        bins = numpy.logspace(numpy.log10(min(abs(frequency[4:]))), numpy.log10(max(abs(frequency))) , bin_size+1)
        weight, mean_power, mean_k = [[] for _ in range(3)]

        loop = 25 if 25 < bin_size else bin_size
        for i in range(bin_size):
            W = numpy.logical_and(norm > bins[i], norm <= bins[i+1]).astype(numpy.int8)
            weight.append(numpy.sum(W))
            if weight[-1] != 0:
                mean_power.append(numpy.sum(W * r.real)/weight[-1])
                mean_k.append(numpy.sum( W * norm)/weight[-1])
            else: mean_power.append(0); mean_k.append(0)

        return {'k': numpy.array(mean_k), 'r': numpy.array(mean_power)}


    def large_scale_bias(self, minmass=0, maxmass=1e20)-> float:
        '''
        This function returns the bias at the largest scales (k < 0.04)
        using the ratio of matter-halo cross-powerspectrum and analytic
        matter powerspectrum.
        '''

        cross_powerspectrum = self.matter_halo_powerspectrum(load=True, minmass=minmass, maxmass=maxmass)
        pmm = self.matter_powerspectrum(load=True)
        k = cross_powerspectrum.k[cross_powerspectrum.k < 0.04]
        bias = numpy.mean((cross_powerspectrum.powerspectrum[k < 0.04]/pmm.powerspectrum[k < 0.04])[1:])
        return bias


    def slice2D(self, include_snapshot=False, percent_width=10):
        """Returns a 2D projected slice from the catalogue.
        Optionally returns that for corresponding snapshot too

        Args:
            include_snapshot (bool, optional): If True, return slice of snapshot too. Defaults to False.
            percent_width (int, optional): Percentage of z axis to be projected upon. Defaults to 10.

        Returns:
            tuple: tuple of sliced array/arrays
        """
        midpoint = int(self.voxels/2)

        catalog = ArrayCatalog({'position': self.position})
        mesh = catalog.to_mesh(BoxSize=self.boxsize, Nmesh=self.voxels, position='position')
        start = midpoint - int(percent_width*self.voxels/100); end = midpoint + int(percent_width*self.voxels/100); 
        logger.info('Computing projected 2D density between grid %s and %s'%(start, end)) 
        slice2d = numpy.array(mesh.paint('real'))[:,:,start:end].sum(2)

        if not include_snapshot:
            return slice2d
        else:
            snapshot = kineticsz.Snapshot(snapshot=self.corresponding_snapshot)
            snapcatalog = ArrayCatalog({'position': snapshot.position})
            mesh = snapcatalog.to_mesh(BoxSize=snapshot.boxsize, Nmesh=self.voxels, position='position')
            snapslice2d = numpy.array(mesh.paint('real'))[:,:,start:end].sum(2)

            return slice2d, snapslice2d



    @property
    def bias(self)-> float:
        """Calculates large scale halo bias based on matter-halo powerspectrum

        Returns:
            float: large scale halo bias
        """
        return self.large_scale_bias()

    def hmf(self, bins=10,  mass_cutoff=0):
        """Calculates and returns halo mass function in units number/mass/volume

        Args:
            bins (int, optional): Number of mass bins to use. Defaults to 10.
            mass_cutoff (int, optional): Minimum halo mass cutoff. Defaults to 0.

        Returns:
            numpy.ndarray: Array containing mass bins and corresponding mass function
        """

        frame = self.frame[self.frame.mvir > mass_cutoff]
        mass = frame.mvir
        number, mass = numpy.histogram(numpy.log10(mass), bins=bins)
        dm = numpy.diff(mass)[0]
    
        halo_mass_function = numpy.column_stack([(mass[1:] + mass[:-1])/2., number/dm/self.boxsize**3])

        return halo_mass_function

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
        density = Density(numpy.array(mesh.paint(mode=mode)*self.nbodykit_factor), voxels=self.voxels, boxsize=self.boxsize, redshift=self.redshift)
        return density



