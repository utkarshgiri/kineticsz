import h5py
import numpy
import pandas
#import pycic
#import powerspectrum


def powerspectrum2D(deltak, boxsize, bins=10):

    bin_size = bins; box_size = boxsize
    grid_size = deltak.shape[0]
    grid_spacing = boxsize / grid_size
    
    delta_k_square = numpy.abs(deltak*deltak)/box_size**2
    kfreq = 2*numpy.pi*numpy.fft.fftfreq(grid_size, grid_spacing)
    kx, ky = numpy.meshgrid(kfreq, kfreq)
    k_norm = knorm = numpy.sqrt(kx**2 + ky**2)[:, :deltak.shape[1]]
    bins = numpy.logspace( numpy.log10(min(abs(kfreq[2:]))), numpy.log10(max(abs(kfreq))) , bin_size+1)
    
    weight, mean_power, mean_k = [[] for _ in range(3)]
    
    for i in range(bin_size):
        W = numpy.logical_and(k_norm > bins[i], k_norm <= bins[i+1]).astype(numpy.int8)
        weight.append(numpy.sum(W))
        if weight[-1] != 0:
            mean_power.append(numpy.sum(W * abs(delta_k_square))/weight[-1])
            mean_k.append(numpy.sum( W * k_norm)/weight[-1])
        else: mean_power.append(0); mean_k.append(0)
    
    return numpy.array(mean_k), numpy.array(mean_power)

def powerspectrum3D(deltak, boxsize, bins=30, deltak2=None):

    if deltak2 is None:
        deltak2 = deltak

    bin_size = bins; box_size = boxsize
    grid_size = deltak.shape[0]
    grid_spacing = boxsize / grid_size

    delta_k_square = numpy.abs(deltak*deltak2) / box_size**3
    kfreq = 2*numpy.pi*numpy.fft.fftfreq(grid_size, grid_spacing).astype(numpy.float32)
    
    k_norm = numpy.zeros((grid_size, grid_size, deltak.shape[2]), dtype=numpy.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            k_norm[i,j,:] = numpy.sqrt(kfreq[i]**2 + kfreq[j]**2 + kfreq[:deltak.shape[2]]**2)
            deltak2[i,j,:] = deltak[i,j,:]*deltak2[i,j,:].conjugate() / box_size**3
    
    
    bins = numpy.logspace( numpy.log10(min(abs(kfreq[4:]))), numpy.log10(max(abs(kfreq))) , bin_size+1)
    #bins = kfreq[:512][1::8]
    weight, mean_power, mean_k = [[] for _ in range(3)]
    
    #loop = 25 if 25 < bin_size else bin_size
    for i in range(len(bins)-1):
        W = numpy.logical_and(k_norm > bins[i], k_norm <= bins[i+1]).astype(numpy.int8)
        weight.append(numpy.sum(W))
        if weight[-1] != 0:
            mean_power.append(numpy.sum(W * abs(deltak2))/weight[-1])
            mean_k.append(numpy.sum( W * k_norm)/weight[-1])
        else: mean_power.append(0); mean_k.append(0)
    
    return numpy.array(mean_k), numpy.array(mean_power)


def velocity_halo(vreconstucted=None, rockstar=None, vk=None, halok=None, boxsize=None, gridsize=None, hubble=0.7, bins=30, mass_cutoff=0):

    if vk is None and halok is None:

        dataframe = pandas.read_csv(rockstar, sep=' ', header = 0, skiprows=19)
        with open(rockstar, "r") as f:
            header = f.readline()
        dataframe.columns = header.split(" ")
        dataframe = dataframe[dataframe['mvir'] > mass_cutoff]
        
        if boxsize is None:
            boxsize = int(numpy.ceil(numpy.max(dataframe.x)/hubble)) #Mpc
        if gridsize is None:
            gridsize = boxsize


        position_halo = numpy.column_stack((dataframe.x, dataframe.y, dataframe.z)).astype(numpy.float64).ravel()/hubble
        mass_halo = numpy.array(dataframe.mvir)

        mean_halo_density = numpy.sum(mass_halo) / boxsize**3
    

        halo_density = pycic.cic(position_halo, mass_halo, boxsize, gridsize)
        halo_density = halo_density/mean_halo_density - 1.
        
        vk = numpy.load(vreconstucted)
        halok = numpy.fft.rfftn(halo_density); del halo_density

        assert vk.shape == halok.shape
    
    else:
        if boxsize is None:
            raise ValueError("No boxsize given")

    bin_size = bins; box_size = boxsize
    grid_size = vk.shape[0]
    grid_spacing = boxsize / grid_size

    delta_k_square = numpy.abs(vk*halok) / box_size**3
    kfreq = 2*numpy.pi*numpy.fft.fftfreq(grid_size, grid_spacing)
    kx, ky, kz = numpy.meshgrid(kfreq, kfreq, kfreq)
    k_norm = knorm = numpy.sqrt(kx**2 + ky**2 + kz**2)[:,:,:int(gridsize/2 + 1)]
    bins = numpy.logspace( numpy.log10(min(abs(kfreq[2:]))), numpy.log10(max(abs(kfreq))) , bin_size+1)

    weight, mean_power, mean_k = [[] for _ in range(3)]

    for i in range(bin_size):
        W = numpy.logical_and(k_norm > bins[i], k_norm <= bins[i+1]).astype(numpy.int8)
        weight.append(numpy.sum(W))
        if weight[-1] != 0:
            mean_power.append(numpy.sum(W * abs(delta_k_square))/weight[-1])
            mean_k.append(numpy.sum( W * k_norm)/weight[-1])
        else: mean_power.append(0); mean_k.append(0)
    
    return numpy.array(mean_k), numpy.array(mean_power)



def matter_halo(snapshot=None, rockstar=None, deltak=None, halok=None, boxsize=None, gridsize=None, hubble=0.7, bins=30, mass_cutoff=0):
    

    if snapshot == rockstar == deltak == halok == None:
        raise ValueError("No argument given to the function")

    if deltak is None and halok is None:

        dataframe = pandas.read_csv(rockstar, sep=' ', header = 0, skiprows=19)
        with open(rockstar, "r") as f:
            header = f.readline()
        dataframe.columns = header.split(" ")
        dataframe = dataframe[dataframe['mvir'] > mass_cutoff]
         
        if boxsize is None:
            boxsize = int(numpy.ceil(numpy.max(position_halo)))
            print('No boxsize supplied to the function. Set boxsize to %s'%boxsize)
        if gridsize is None:
            gridsize = boxsize
            print('No boxsize supplied to the function. Set gridsize to %s'%boxsize)

        dm_positions = h5py.File(snapshot, 'r')['PartType1']['Coordinates'][:].astype(numpy.float64)
        halo_positions = numpy.column_stack((dataframe.x, dataframe.y, dataframe.z)).astype(numpy.float64)/hubble
        halo_masses = numpy.array(dataframe.mvir)

        mean_halo_density = halo_positions.shape[0] / boxsize**3
        mean_number_density = dm_positions.shape[0] / boxsize**3
    
        deltak = numpy.fft.rfftn(pycic.cic(dm_positions.ravel(), numpy.ones(dm_positions.shape[0]), boxsize, gridsize))/mean_number_density

        halok = numpy.fft.rfftn(pycic.cic(halo_positions.ravel(), numpy.ones_like(halo_masses), boxsize, gridsize))/mean_halonumber_density
        
   
    else:
        if boxsize is None:
            raise ValueError("No boxsize given")



    return powerspectrum3D(deltak=deltak, deltak2=halok, boxsize=int(boxsize), bins=bins)
    

def halo_halo(rockstar=None, halok=None, boxsize=None, gridsize=None, bins=30, hubble=0.7, mass_cutoff=0):
    
    print('Using %s bins to perform gridding'%bins)

    if rockstar == halok == None:
        raise ValueError("No rockstar filename or halo density field given to the function")

    if halok is None:

        dataframe = pandas.read_csv(rockstar, sep=' ', header = 0, skiprows=19)
        with open(rockstar, "r") as f:
            header = f.readline()
        dataframe.columns = header.split(" ")
        dataframe = dataframe[dataframe['mvir'] > mass_cutoff]
    
        halo_positions = numpy.column_stack((dataframe.x, dataframe.y, dataframe.z)).astype(numpy.float64)/hubble
        halo_masses = numpy.array(dataframe.mvir)
        
        if boxsize is None:
            boxsize = int(numpy.ceil(numpy.max(position_halo)))
            print('No boxsize supplied to the function. Set boxsize to %s'%boxsize)
        if gridsize is None:
            gridsize = boxsize
            print('No boxsize supplied to the function. Set gridsize to %s'%boxsize)

        mean_number_density = halo_positions.shape[0]/boxsize**3 

        halok = numpy.fft.rfftn(pycic.cic(halo_positions.ravel(), numpy.ones_like(halo_masses), boxsize, gridsize))/mean_number_density
   
    else:
        if boxsize is None:
            raise ValueError("No boxsize given")


    return powerspectrum3D(deltak=halok, boxsize=boxsize, bins=bins)


def matter_matter(snapshot=None, deltak=None, boxsize=None, gridsize=None, bins=30):

    if snapshot == deltak == None:
        raise ValueError("No argument given to the function")

    if deltak is None:

        boxsize = int(list(h5py.File(snapshot, 'r')['Header'].attrs.items())[0][1]) #Mpc
        if gridsize is None:
            gridsize = boxsize

        position_snap = h5py.File(snapshot, 'r')['PartType1']['Coordinates'][:].astype(numpy.float64).ravel()

        mean_dm_density = float(len(position_snap)/3) / boxsize**3
    
        dm_density = pycic.cic(position_snap, numpy.ones(int(len(position_snap)/3)), boxsize, gridsize)

        deltak = numpy.fft.fftn(dm_density)
   
    else:
        if boxsize is None:
            raise ValueError("No boxsize given")


    return powerspectrum3D(deltak=deltak, boxsize=boxsize, bins=bins)
 

def bias(snapshot=None, rockstar=None):
    pass




def matter_halo_bias(snapshot=None, rockstar=None, deltak=None, halok=None, boxsize=None, gridsize=None, hubble=0.7, bins=30, mass_cutoff=0):

    if snapshot == rockstar == deltak == halok == None:
        raise ValueError("No argument given to the function")

    if deltak is None and halok is None:

        dataframe = pandas.read_csv(rockstar, sep=' ', header = 0, skiprows=19)
        with open(rockstar, "r") as f:
            header = f.readline()
        dataframe.columns = header.split(" ")
        dataframe = dataframe[dataframe['mvir'] > mass_cutoff]
        
        if boxsize is None:
            boxsize = int(list(h5py.File(snapshot, 'r')['Header'].attrs.items())[0][1]) #Mpc
        if gridsize is None:
            gridsize = boxsize


        position_snap = h5py.File(snapshot, 'r')['PartType1']['Coordinates'][:].astype(numpy.float64).ravel()
        position_halo = numpy.column_stack((dataframe.x, dataframe.y, dataframe.z)).astype(numpy.float64).ravel()/hubble
        mass_halo = numpy.array(dataframe.mvir)

        mean_halo_density = numpy.sum(mass_halo) / boxsize**3
        mean_dm_density = float(len(position_snap)/3) / boxsize**3
    
        dm_density = pycic.cic(position_snap, numpy.ones(int(len(position_snap)/3)), boxsize, gridsize)
        #dm_density = dm_density/mean_dm_density - 1.

        halo_density = pycic.cic(position_halo, numpy.ones_like(mass_halo), boxsize, gridsize)
        #halo_density = halo_density/mean_halo_density - 1.
        
        del position_snap; del position_halo;
        deltak = numpy.fft.fftn(dm_density); del dm_density
        halok = numpy.fft.fftn(halo_density); del halo_density
    

        delta = halok/deltak
   
    else:
        if boxsize is None:
            raise ValueError("No boxsize given")

    k, biassquare = powerspectrum3D(deltak=delta, boxsize=int(boxsize), bins=bins)
    return k, numpy.sqrt(biassquare)
    

