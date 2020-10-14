import numpy

def modes(array, boxsize, kmax=0.1, mu_min=0.0, mu_max=1.0):
    '''
    Args:
    
    array: 3D array to be cropped
    boxsize: size of tthe box to which the array belongs
    kmax: maximum value of k to go to
    mu_min: minimum value of kz/k to keep
    mu_max: maximum value of kz/k to keep
    
    Returns:

    kx, ky, kz, k, array : 1D vectors of kx, ky, kz, k and array
    '''

    gridsize = array.shape[0]
    frequency = 2*numpy.pi*numpy.fft.fftfreq(gridsize, boxsize/gridsize)
    
    kx, ky, kz = numpy.meshgrid(frequency, frequency, frequency[:array.shape[2]])
    norm = numpy.sqrt(kx**2 + ky**2 + kz**2)
    norm[0,0,0] = 1.0

    k = norm.flatten(); array = array.flatten(); 
    kx = kx.flatten(); ky = ky.flatten(); kz = kz.flatten()

    args = numpy.argsort(k)
    sortedk = k[args]

    if type(kmax) is int and kmax >= 1:
        args = args[:kmax]
    else:
        args = args[sortedk < kmax]

    k = k[args]; kx = kx[args]; ky = ky[args]; kz = kz[args]
    array = array[args]

    if mu_min or mu_max != 1:
        mu = kz/numpy.sqrt(kx**2 + ky**2 + kz**2)
        args = numpy.where((mu_min < numpy.abs(mu)) & (numpy.abs(mu) < mu_max))
        k = k[args]; kx = kx[args]; ky = ky[args]; kz = kz[args]
        array = array[args]

    return kx[1:], ky[1:], kz[1:], k[1:], array[1:]

def modes_and_indices(array, boxsize, kmax=0.5, mu_min=0.0, mu_max=1.0):
    '''
    Args:
    
    array: 3D array to be cropped
    boxsize: size of tthe box to which the array belongs
    kmax: maximum value of k to go to
    mu_min: minimum value of kz/k to keep
    mu_max: maximum value of kz/k to keep
    
    Returns:

    kx, ky, kz, k, array : 1D vectors of kx, ky, kz, k and array
    '''

    gridsize = array.shape[0]
    frequency = 2*numpy.pi*numpy.fft.fftfreq(gridsize, boxsize/gridsize)
    
    kx, ky, kz = numpy.meshgrid(frequency, frequency, frequency[:array.shape[2]])
    norm = numpy.sqrt(kx**2 + ky**2 + kz**2)
    norm[0,0,0] = 1.0

    k = norm.flatten(); array = array.flatten(); 
    kx = kx.flatten(); ky = ky.flatten(); kz = kz.flatten()

    args = numpy.argsort(k)
    sortedk = k[args]

    if type(kmax) is int and kmax >= 1:
        args = args[:kmax]
    else:
        args = args[sortedk < kmax]

    k = k[args]; kx = kx[args]; ky = ky[args]; kz = kz[args]
    array = array[args]

    return kx, ky, kz, k, array, args
