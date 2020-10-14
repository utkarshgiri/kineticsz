import fire
import json
import numpy
import slurmpy
from rich import print
from pathlib import Path

def main(paramfile, partition='defq', nodes=1):
    
    with open(paramfile, 'r') as f:
        lines = f.readlines()
    
    dictionary = dict()
    
    for line in lines:
        if line[0] in ('%', '\n', '', ' '):
            continue
        else:
            splits = line.split()
            key, value = splits[0], splits[1]
            dictionary[key] = value 

    class_paramfile = Path(dictionary['InitCondFile']).parent.joinpath('initial_configuration.json')
    
    with open(class_paramfile, 'r') as f:
        config = json.load(f)
    
    classomega0 = (float(config['omega_b']) + float(config['omega_cdm']))/float(config['h'])**2
    classomegab = float(config['omega_b'])/float(config['h'])**2

    assert numpy.isclose(float(dictionary['Omega0']), classomega0, atol=1e-4)
    assert numpy.isclose(float(dictionary['OmegaLambda']), 1 - float(dictionary['Omega0']), atol=1e-4)
    assert numpy.isclose(float(dictionary['OmegaBaryon']), classomegab, atol=1e-4)
    assert numpy.isclose(float(dictionary['HubbleParam']), float(config['h']), atol=1e-4)
    assert numpy.isclose(float(dictionary['BoxSize']), float(config['boxsize']), atol=1e-4)
    assert numpy.isclose(float(dictionary['TimeBegin']), 1./( 1. + float(config['z_pk'])), atol=1e-4)

    s = slurmpy.Slurm('Gadget', {'partition': partition,
                                     'time': '23:55:55',
                                     'nodes': nodes})

    s.run('export LD_LIBRARY_PATH=/home/ugiri/installations/lib/hdf5/1.6.9/lib/:/home/ugiri/installations/lib/gsl/1.9/lib:$LD_LIBRARY_PATH && \
           /cm/shared/apps/openmpi/gcc/64/1.10.7/bin/mpiexec -n {} /home/ugiri/projects/Gadget/Gadget/Gadget2 {}'.format(int(nodes)*40, paramfile))


if __name__ == '__main__':
    fire.Fire(main)
