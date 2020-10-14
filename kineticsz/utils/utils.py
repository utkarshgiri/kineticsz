import numpy
import kineticsz
from pathlib import Path, PosixPath


def find_snapshot_and_catalog(directory, redshift: int=2,  meta: str=''):
    """A helper function which finds the snapshot file and rockstar catalog for a given directory name
    Arguments:
        directory {Path} -- path of the directory
    Keyword Arguments:
        meta {str} -- Optional metadata (default: {''})
    Returns:
        [type] -- Snapshot and Catalog objects
    """
    
    if not isinstance(directory, PosixPath):
        directory = Path(directory)
    
    snapshots = list(directory.glob('*.hdf5'))
    catalog = directory.joinpath('halos_0.0.ascii')

    if not catalog.exists():
        raise Exception('Catalog not found in the {} directory'.format(meta))
    
    rockstar = kineticsz.Rockstar(rockstar=catalog.as_posix())
    assert numpy.isclose(rockstar.redshift, redshift, atol=0.01)

    for name in snapshots:
        snapshot = kineticsz.Snapshot(name)
        if not numpy.isclose(snapshot.redshift, redshift, atol=0.01):
            snapshots.remove(name)

    return snapshots, catalog.as_posix()


def k3D(boxsize, voxels):
    k = 2*numpy.pi*numpy.fft.fftfreq(n=voxels, d=boxsize/voxels)
    k3d = numpy.zeros((voxels, voxels, int(voxels/2 + 1)))
    for i in range(voxels):
        for j in range(voxels):
            k3d[i,j,:] = numpy.sqrt(k[i]**2 + k[j]**2 + k[:int(voxels/2 +1)]**2)
    return k3d

def k2D(boxsize, voxels):
    k = 2*numpy.pi*numpy.fft.fftfreq(n=voxels, d=boxsize/voxels)
    k2d = numpy.zeros((voxels, int(voxels/2 + 1)))
    for i in range(voxels):
        k2d[i,:] = numpy.sqrt(k[i]**2 + k[:int(voxels/2 +1)]**2)
    return k2d
