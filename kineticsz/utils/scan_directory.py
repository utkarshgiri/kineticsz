from kineticsz import read
from pathlib import Path, PosixPath


def find_snapshot_and_catalog(directory, redshift: int=2,  meta: str=''):
    """A helper function which finds the snapshot file and rockstar catalog for a given directtory name
    Arguments:
        directory {Path} -- path of the directory
    Keyword Arguments:
        meta {str} -- Optional metadata (default: {''})
    Returns:
        [type] -- Snapshot and Catalog objects
    """
    if not isinstance(directory, pathlib.PosixPath):
        directory = Path(directory)
    
    snapshots = directory.glob('snapshot*.hdf5')
    catalogs = directory.glob('halos*')

    snapshot, rockstar = None, None

    for name in catalogs:
        rockstar = read.Rockstar(rockstar=name)
        if redshift - 0.01 < rockstar.redshift < redshift + 0.01:
            break
    
    for name in snapshots:
        snapshot = read.Snapshot(name)
        if (redshift - 0.01 < snapshot.redshift < redshift + 0.01):
            break

    return snapshot, rockstar


