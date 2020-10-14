import subprocess
from pathlib import Path
import shutil
import sys
import time
dirname = sys.argv[1]
#for dirname in range(10,30):
#destinations = sorted(list(Path('/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/').glob('[7*')))
#print(destinations)
#for dirname in list(reversed(destinations))[:8]:
    #destination = f'/scratch/ugiri/quijote/{dirname}/snapdir_001'
destination = f'/gpfs/ugiri/Quijote/scratch/ugiri/Quijote/{dirname}'
#if Path(destination).joinpath('snapdir_001/halos_0.0.ascii').exists(): sys.exit()
filenames = sorted(list(Path(destination).joinpath('snapdir_001').glob('*.hdf5')))
if len(filenames) != 8: sys.exit()
subprocess.call(['/home/ugiri/installations/rockstar/rockstar', '-c', '/home/ugiri/installations/rockstar/quickstart_quixote.cfg'] + filenames)
shutil.move('halos_0.0.ascii', destination + '/snapdir_001/halos_0.0.ascii')
