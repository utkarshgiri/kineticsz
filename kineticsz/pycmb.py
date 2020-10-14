import sys
import copy
import numpy
import kineticsz
from box import Box
from pathlib import Path
from classy import Class
from typing import Tuple
from nbodykit.lab import *
from scipy import constants, interpolate
from kineticsz.utils import pypowerspectrum, k2D
numpy.random.seed(42)

class CMB():

    def __init__(self, parameters=None, verbose=False, units='mpc'):
        #if no parameters given, use default parameter
        if parameters is None: parameters = self.default_parameters()
        #unwrap parameter into a classy compatible dictionary
        self.parameters = self.unwrap_class_params(parameters)
        #Instantiating classy object
        self.model = Class()
        self.model.set(self.parameters)
        self.model.compute()
        #storing redshift of the classy instance
        self.redshift = self.parameters['z_pk']
        if units == 'mpch':
            self.factor = 1/self.parameters['h']
            self.hfactor = self.parameters['h']
        else:
            self.factor = 1
            self.hfactor = 1

        self.arcmin_to_radian = numpy.pi/180./60.
        self.radian_to_arcmin = 1./(self.arcmin_to_radian)

    def default_parameters(self)-> dict:
        """This function reads in and returns the default parameter used for classy
        Returns:
            [dict] -- A dictionary of classy readable/compatible parameter
        """
        return '../ics/configuration/class_params.json'

    @property
    def Kstar(self)-> float:
        """The coefficient which appears before the integral term in kSZ
        Returns:
            float -- The coefficient
        """
        mtompc = 3.24078e-23
        rho_critical = 1e-26/mtompc**3
        omega_b = 0.04
        n_e0 = (rho_critical * omega_b) / (1.14*constants.proton_mass)
        sigmaT = constants.physical_constants['Thomson cross section'][0]*mtompc**2
        x_e = 1.0; exptau = 1.0
        speed_of_light = constants.speed_of_light/1e3
        K_star = self.Tcmb * n_e0 * sigmaT * x_e * exptau / speed_of_light
        return K_star/self.hfactor

    def Kksz(self, redshift):
        return self.Kstar*(1+redshift)**2

    @property
    def Tcmb(self)-> float:
        """The CMB temperature in micro kelvin
        Returns:
            float -- CMB temperature
        """
        return 2.72e6

    @property
    def c(self)-> float:
        """Speed of light in km/s
        Returns:
            float -- Speed
        """

        return constants.speed_of_light/1e3

    def Chistar(self, redshift:float)-> float:
        """Computes comoving distance to the given redshift
        Arguments:
            redshift {float} -- The redshift at which the distance is desired
        Returns:
            float -- comoving distance
        """
        #an object storing the classy background
        bg = self.model.get_background()
        #create an interpolator object
        chi_star_interpolator = interpolate.interp1d(bg['z'][:], bg['comov. dist.'][:])
        #interpolate comoving distance
        chi_star = chi_star_interpolator(redshift)
        return float(chi_star*self.hfactor)


    def unwrap_class_params(self, parameters):
        """unwrap the contents of `parameters` into a dictionary
        Arguments:
            parameters {[type]} -- A dictionary containing the 
        Raises:
            TypeError: [description]
        Returns:
            [type] -- A dictionary of class parameters
        """
        #if the parameters type is `str`
        if type(parameters) == str:
            #assert that its a name of file on disk
            assert Path(parameters).exists()
            #assert that the file has a suffix `npy`
            if parameters.endswith('.npy'):
                parameters = numpy.load(parameters, allow_pickle=True).item()
            elif parameters.endswith('.json'):
                import json
                with open(parameters, 'r') as f:
                    parameters = json.load(f)
            else:
                print('parameter file must be of either .json or .npy format')
        elif type(parameters) == dict:
            parameters = parameters
        else:
            raise TypeError

        return parameters

    def Cl(self, lmax :int=2000, scale_invariant :bool=False, lensing :bool=False)-> Box:
        """Function to compute CMB Cls
        Keyword Arguments:
            lmax {int} -- multipole upto which Cls are to be computed (default: {2000})
            scale_invariant {bool} -- If True, return scale invariant Cls (default: {False})
            lensing {bool} -- If True, return lensed Cls (default: {False})
        Returns:
            [Box] -- Box 
        """
        try:
            if lensing: cl = self.model.lensed_cl(lmax=lmax)
            else: cl = self.model.raw_cl(lmax=lmax)
        except:
            parameters = copy.deepcopy(self.parameters)
            parameters['output'] = 'lCl tCl mPk'
            parameters['l_max_scalars'] = lmax + 2000
            if lensing: parameters['lensing'] = 'yes'
            model = Class()
            model.set(parameters)
            model.compute()
            if lensing: cl = model.lensed_cl(lmax=lmax)
            else: cl = model.raw_cl(lmax=lmax)
        el = cl['ell'][2:]
        Cl = self.Tcmb**2*cl['tt'][2:]
        if scale_invariant: Cl = el*(el+1)*Cl/(2*numpy.pi)
        return Box({'ell': el, 'cl': Cl})


    def deflection_cl(self, lmax :int=2000, scale_invariant :bool=False)-> Box:
        """Function to compute deflection field Cls
        Keyword Arguments:
            lmax {int} -- multipole upto which Cls are to be computed (default: {2000})
            scale_invariant {bool} -- If True, return scale invariant Cls (default: {False})
            lensing {bool} -- If True, return lensed Cls (default: {False})
        Returns:
            [Box] -- Box 
        """
        parameters = copy.deepcopy(self.parameters)
        parameters['output'] = 'lCl tCl mPk'
        parameters['l_max_scalars'] = lmax + 2000
        model = Class()
        model.set(parameters)
        model.compute()
        cl = model.raw_cl(lmax=lmax)
        el = cl['ell'][2:]
        lensingcl = cl['pp'][2:]
        if scale_invariant: deflectioncl = el*(el+1)*lensingcl
        return Box({'ell': el, 'cl': deflectioncl})


    def lensing_potential_cl(self, lmax :int=2000, scale_invariant :bool=False)-> Box:
        """Function to compute lensing potential Cls
        Keyword Arguments:
            lmax {int} -- multipole upto which Cls are to be computed (default: {2000})
            scale_invariant {bool} -- If True, return scale invariant Cls (default: {False})
            lensing {bool} -- If True, return lensed Cls (default: {False})
        Returns:
            [Box] -- Box 
        """
        parameters = copy.deepcopy(self.parameters)
        parameters['output'] = 'lCl tCl mPk'
        parameters['l_max_scalars'] = lmax + 2000
        model = Class()
        model.set(parameters)
        model.compute()
        cl = model.raw_cl(lmax=lmax)
        el = cl['ell'][2:]
        lensingcl = cl['pp'][2:]
        if scale_invariant: lensingcl = el*(el+1)*lensingcl/(2*numpy.pi)
        return Box({'ell': el, 'cl': lensingcl})


    def lensed_Cl(self, lmax: int=2000, scale_invariant: bool=False, lensing: bool=True)-> Box:
        """A function which computes lensed Cls
        Keyword Arguments:
            lmax {int} -- mulptipole upto which Cls are to be computed (default: {2000})
            scale_invariant {bool} -- if True, return scale invariant Cls (default: {False})
            lensing {bool} -- [description] (default: {True})
        Returns:
            [Box] -- Box object containg ells and Cls
        """
        return self.Cl(lmax=lmax, scale_invariant=scale_invariant, lensing=lensing)


    def powerspectrum(self, lmax: int=2000, scale_invariant :bool=True, lensing: bool=False):
        """Returns Box type object containing CMB powerspectrum i.e. ell and cl
        Keyword Arguments:
            lmax {int} -- value of l upto which Cl is desired (default: {2000})
            scale_invariant {bool} -- [description] (default: {True})
            lensing {bool} -- [description] (default: {False})
        Returns:
            [type] -- [description]
        """
        if lensing:
            return self.lensed_Cl(lmax, scale_invariant)
        else:
            return self.Cl(lmax, scale_invariant)



    def deflection2D(self, redshift: float,  boxsize: float=512, voxels: int=None, mode :str='real', lensed_powerspectrum=False)-> numpy.ndarray:
        """Function creates a 2D CMB map
        Arguments:
            redshift {float} -- redshift at which the map is to be computed. This determines the angular coordinate
        Keyword Arguments:
            boxsize {float} -- Size of map (default: {512})
            voxels {int} -- Number of pixel elements to be used (default: {1024})
            mode {str} -- whether to return real or complex map (default: {'real'})
            lensed_powerspectrum {bool} -- whether to use lensed CMB power (default: {False})
        Returns:
            numpy.ndarray -- A 2D array containing the map
        """
        if voxels is None:
            voxels = min(int(boxsize), 1024)
        gridding_factor = boxsize/voxels
        chi_star = self.Chistar(redshift)
        freq = 2*numpy.pi*numpy.fft.fftfreq(voxels, gridding_factor)
        kx, ky = numpy.meshgrid(freq, freq)
        norm = numpy.sqrt(kx**2 + ky**2)[:,:int(len(freq)/2 + 1)]*chi_star
        powerspectrum = self.deflection_cl(lmax=14000, scale_invariant=False)

        cosmo, el, Cl = self.model, powerspectrum.ell, powerspectrum.cl
        Cl = (Cl*(boxsize*chi_star)**2)/2.0

        std2d = numpy.sqrt(numpy.interp(norm, el, Cl))
        map2d = numpy.random.normal(scale=std2d) + 1j*numpy.random.normal(scale=std2d)

        if mode == 'real':
            return numpy.fft.irfft2(map2d).real
        else:
            return map2d


    def lensing_potential2D(self, redshift: float,  boxsize: float=512, voxels: int=None, mode :str='real', lensed_powerspectrum=False)-> numpy.ndarray:
        """Function creates a 2D lensing potential map
        Arguments:
            redshift {float} -- redshift at which the map is to be computed. This determines the angular coordinate
        Keyword Arguments:
            boxsize {float} -- Size of map (default: {512})
            voxels {int} -- Number of pixel elements to be used (default: {1024})
            mode {str} -- whether to return real or complex map (default: {'real'})
            lensed_powerspectrum {bool} -- whether to use lensed CMB power (default: {False})
        Returns:
            numpy.ndarray -- A 2D array containing the map
        """
        if voxels is None:
            voxels = min(int(boxsize), 1024)
        gridding_factor = boxsize/voxels
        chi_star = self.Chistar(redshift)
        freq = 2*numpy.pi*numpy.fft.fftfreq(voxels, gridding_factor)
        kx, ky = numpy.meshgrid(freq, freq)
        norm = numpy.sqrt(kx**2 + ky**2)[:,:int(len(freq)/2 + 1)]*chi_star
        powerspectrum = self.lensing_potential_cl(lmax=30000, scale_invariant=False)

        cosmo, el, Cl = self.model, powerspectrum.ell, powerspectrum.cl
        Cl = (Cl*(boxsize*chi_star)**2)/2.0

        std2d = numpy.sqrt(numpy.interp(norm, el, Cl))
        map2d = numpy.random.normal(scale=std2d) + 1j*numpy.random.normal(scale=std2d)

        if mode == 'real':
            return numpy.fft.irfft2(map2d).real
        else:
            return map2d



    def cmbmap2D(self, redshift: float, boxsize: float=512, voxels: int=None, mode :str='real', lensed_powerspectrum=False)-> numpy.ndarray:
        """Function creates a 2D CMB map
        Arguments:
            redshift {float} -- redshift at which the map is to be computed. This determines the angular coordinate
        Keyword Arguments:
            boxsize {float} -- Size of map (default: {512})
            voxels {int} -- Number of pixel elements to be used (default: {1024})
            mode {str} -- whether to return real or complex map (default: {'real'})
            lensed_powerspectrum {bool} -- whether to use lensed CMB power (default: {False})
        Returns:
            numpy.ndarray -- A 2D array containing the map
        """
        if voxels is None:
            voxels = min(int(boxsize), 1024)
        gridding_factor = boxsize/voxels
        chi_star = self.Chistar(redshift)
        freq = 2*numpy.pi*numpy.fft.fftfreq(voxels, gridding_factor)
        kx, ky = numpy.meshgrid(freq, freq)
        norm = numpy.sqrt(kx**2 + ky**2)[:,:int(len(freq)/2 + 1)]*chi_star
        if lensed_powerspectrum:
            powerspectrum = self.powerspectrum(lmax=14000, lensing=lensed_powerspectrum, scale_invariant=False)
        else:
            powerspectrum = self.powerspectrum(lmax=14000, scale_invariant=False)

        cosmo, el, Cl = self.model, powerspectrum.ell, powerspectrum.cl
        Clcmb = (Cl*(boxsize*chi_star)**2)/2.0

        std2d = numpy.sqrt(numpy.interp(norm, el, Clcmb))
        map2d = numpy.random.normal(scale=std2d) + 1j*numpy.random.normal(scale=std2d)

        if mode == 'real':
            return numpy.fft.irfft2(map2d).real
        else:
            return map2d

    def kszmap2D(self, snapshot, voxels: int=None, mode: str='real')-> numpy.ndarray:
        """A function to produce 2D kSZ map from a given simulation
        Arguments:
            snapshot {kineticsz.Snapshot} -- A Snapshot type object for which kSZ map is desired
        Keyword Arguments:
            voxels {int} -- Number of voxels to use for the map (default: {None})
            mode {str} -- Whether to produce a map in real or complex space (default: {'real'})
        Returns:
            [numpy.ndarray] -- A 2D array containing the map
        """
        if voxels is None:
            voxels = min(int(snapshot.boxsize), 1024)
        position = snapshot.position()
        velocity = snapshot.comoving_vz()
        catalog = ArrayCatalog({'position': position, 'velocity': velocity})
        map2d = catalog.to_mesh(BoxSize=int(snapshot.boxsize), Nmesh=voxels, position='position', value='velocity')
        map2d = numpy.array(map2d.paint(mode='real')).sum(2)
        map2d = (self.Kstar * (1+snapshot.redshift)**2) * map2d #/ self.Chistar(snapshot.redshift)**2
        numpy.save(Path(snapshot.dirname).joinpath('kSZmap_%s.npy'%snapshot.tag), map2d)

        if mode == 'real':
            return map2d
        else:
            return numpy.fft.rfft2(map2d)

    def kszck_from_simulation(self, snapshot, voxels=None, bins=100) -> Box:
        """Compute kSZ powerspectrum
        Arguments:
            snapshot {kineticsz.Snapshot} -- A Snapshot type object for which kSZ map is desired
        Keyword Arguments:
            voxels {int} -- Number of voxels to use for the map (default: {None})
            bins {int} -- Number of bins for binning (default: {20})
        Returns:
            Box[numpy.ndarray, numpy.ndarray] -- Box type object containg ks and Pks
        """
        momentum_powerspectrum = snapshot.momentum_powerspectrum(voxels=voxels)
        momentum_powerspectrum.powerspectrum = momentum_powerspectrum.powerspectrum[:]*(self.Kstar*(1+snapshot.redshift)**2)**2*snapshot.boxsize
        #map2D = self.kszmap2D(snapshot, voxels=voxels)
        #k, Pk = pypowerspectrum.powerspectrum2D(numpy.fft.rfft2(map2D), boxsize=snapshot.boxsize, bins=bins)
        #Pk = numpy.interp(k, momentum_powerspectrum.k, momentum_powerspectrum.powerspectrum)
        #return momentum_powerspectrum #
        return Box({'k': momentum_powerspectrum.k, 'powerspectrum': momentum_powerspectrum.powerspectrum})


    def kszclass(self, snapshot, use_simulation_for_variance=False):
        if use_simulation_for_variance:
            variance = numpy.var(snapshot.peculiar_vz)
        else:
            variance = self.velocity_variance(kmin=snapshot.fundamental_mode, kmax=snapshot.nyquist, redshift=snapshot.redshift)/3.
        print(variance)
        chi_star = self.Chistar(snapshot.redshift)
        def Pqq(l):
            return variance * self.model.pk(l/chi_star, snapshot.redshift)
        elksz = numpy.arange(1000,16000)
        Cl = []
        for l in elksz:
            Cl.append(snapshot.boxsize * self.Kstar**2 * (1+snapshot.redshift)**4 * Pqq(l) / chi_star**2)
        #Dlkszclass = elksz * (elksz + 1) * numpy.array(Cl) / (2 * numpy.pi)
        Dlkszclass = numpy.array(Cl)
        return elksz, Dlkszclass


    def velocity_variance(self, redshift=2., kmin=1e-3, kmax=5e1):

        k = numpy.logspace(start=numpy.log10(kmin), stop=numpy.log10(kmax), num=10000) #unit=1/Mpc

        mpower, vpower = [numpy.zeros_like(k) for _ in range(2)]

        bg = self.model.get_background()

        f = interpolate.interp1d(bg['z'][:], bg['gr.fac. f'][:])
        c = 3e5
        hz = self.model.Hubble(redshift)*c / 100. #unit=km/(s.Mpc)

        for i, kvalue in enumerate(k):
            faH = f(redshift)*(1./(1.+redshift))*hz*100/kvalue #unit=km/s
            mpower[i] = self.model.pk(kvalue, redshift)             #unit=Mpc^3
            vpower[i] = mpower[i] * faH**2                     #unit=(km/s)^2.Mpc^3


        vpower_in_mpc_units = vpower / c**2 #unit=Mpc^3

        variance = numpy.trapz(k**2*vpower/(2*numpy.pi**2), k) #unitless

        return variance


    def cmbck_from_simulation(self, snapshot, boxsize=None, voxels=None, bins=20) -> Tuple[numpy.ndarray, numpy.ndarray]:
        '''
        This function returns kSZ powerspectrum a function of k
        '''
        if boxsize is None:
            boxsize = snapshot.boxsize
        map2D = self.cmbmap2D(redshift=snapshot.redshift, boxsize=boxsize)
        k, Pk = pypowerspectrum.powerspectrum2D(numpy.fft.rfft2(map2D), boxsize=boxsize, bins=bins)
        return Box({'k': k, 'powerspectrum': Pk})


    def cmbcl_from_simulation(self, redshift=2.0, lmin=500, lmax=5000, bins=20, boxsize=512, voxels=512, scale_invariant=True):
        '''
        Returns cmb powerspectrum as a function of multiplole l computed from simulation of cmb map

        Args:
            redshift (float) :  redshift where the map is placed. Defaults to 2.
            lmin (int) : minimum value of l for which ksz powerspectrum is computed. Defaults to 500
            lmax (int) : maximum value of l for which ksz powerspectrum is computed. Defaults to 5000
            bins (int) : Number of bins in which averaging of powerspectrum is performed. Defaults to 20
            voxels (int) : Number of voxels to perform CIC. Defaults to 1024
        Returns:
            A box object containg l, Cl
        '''
        map2d = self.cmbmap2D(redshift, boxsize, voxels)
        angular_boxsize = (boxsize/self.Chistar(redshift))
        elcmb2d, Clcmb2d = pypowerspectrum.powerspectrum2D(numpy.fft.fft2(map2d), boxsize=angular_boxsize, bins=bins)
        Clcmb2d = Clcmb2d[(elcmb2d > lmin) & (elcmb2d < lmax)]
        elcmb2d = elcmb2d[(elcmb2d > lmin) & (elcmb2d < lmax)]

        return Box({'ell': elcmb2d, 'cl': elcmb2d*(elcmb2d+1)*Clcmb2d/(2*numpy.pi)})

    def kszcl_from_simulation(self, snapshot, lmin=500, lmax=5000, bins=20, voxels=1024):
        '''
        Returns ksz powerspectrum as a function of multiplole l

        Args:
            snapshot (kineticsz.Snapshot) : snapshot to use for ksz map 
            lmin (int) : minimum value of l for which ksz powerspectrum is computed. Defaults to 500
            lmax (int) : maximum value of l for which ksz powerspectrum is computed. Defaults to 5000
            bins (int) : Number of bins in which averaging of powerspectrum is performed. Defaults to 20
            voxels (int) : Number of voxels to perform CIC. Defaults to 1024
        Returns:
            A box object containg l, Cl
        '''
        map2D = self.kszmap2D(snapshot, voxels)
        angular_boxsize = (snapshot.boxsize/self.Chistar(snapshot.redshift))
        elksz2d, Clksz2d = pypowerspectrum.powerspectrum2D(numpy.fft.fft2(map2D), boxsize=angular_boxsize, bins=bins)
        Clksz2d = Clksz2d[(elksz2d > lmin) & (elksz2d < lmax)]
        elksz2d = elksz2d[(elksz2d > lmin) & (elksz2d < lmax)]

        return Box({'ell': elksz2d, 'cl': elksz2d*(elksz2d+1)*Clksz2d/(2*numpy.pi)})

    def interpolate_matter_powerspectrum(self, k=None, redshift=None, careful_interpolation=False):
        '''
        Returns a matter powerspectrum interpolation object

        Args:
            k (numpy.ndarray): array of k values to use for performing interpolation
            redshift (float): Redshift to which the transfer function is evolved to
            careful_interpolation (bool): Parameter for careful extrapolation.

        Returns:
            An interpolation object
        '''
        if k is None:
            k = numpy.geomspace(1e-4,1e2, 5000)
        if redshift is None:
            redshift = self.redshift
        powerspectrum = numpy.zeros_like(k)

        for i,kval in enumerate(k):
            powerspectrum[i] = self.model.pk(kval/self.factor, redshift)/self.factor**3

        interpolation = interpolate.interp1d(k, powerspectrum, fill_value='extrapolate', bounds_error=careful_interpolation)

        return interpolation


    def interpolate_linear_powerspectrum(self, k=None, redshift=None, careful_interpolation=False):
        '''
        Returns a matter powerspectrum interpolation object

        Args:
            k (numpy.ndarray): array of k values to use for performing interpolation
            redshift (float): Redshift to which the transfer function is evolved to
            careful_interpolation (bool): Parameter for careful extrapolation.

        Returns:
            An interpolation object
        '''
        parameters = self.parameters.copy()
        if k is None:
            k = numpy.geomspace(1e-4,4e2, 5000)
        if redshift is None:
            redshift = self.redshift
        try:
            parameters.pop('non linear')
        except:
            pass
        model = Class()
        model.set(parameters)
        model.compute()
        powerspectrum = numpy.zeros_like(k)

        for i,kval in enumerate(k):
            powerspectrum[i] = model.pk(kval/self.factor, redshift)/self.factor**3

        interpolation = interpolate.interp1d(k, powerspectrum, fill_value='extrapolate', bounds_error=careful_interpolation)

        return interpolation


    def interpolate_nonlinear_powerspectrum(self, k=None, redshift=None, careful_interpolation=False):
        '''
        Returns a matter powerspectrum interpolation object

        Args:
            k (numpy.ndarray): array of k values to use for performing interpolation
            redshift (float): Redshift to which the transfer function is evolved to
            careful_interpolation (bool): Parameter for careful extrapolation.

        Returns:
            An interpolation object
        '''
        parameters = self.parameters.copy()
        if k is None:
            k = numpy.geomspace(1e-4,4e2, 5000)
        if redshift is None:
            redshift = self.redshift
        #parameters.pop('non linear')
        model = Class()
        model.set(parameters)
        model.compute()
        powerspectrum = numpy.zeros_like(k)

        for i,kval in enumerate(k):
            powerspectrum[i] = model.pk(kval/self.factor, redshift)/self.factor**3

        interpolation = interpolate.interp1d(k, powerspectrum, fill_value='extrapolate', bounds_error=careful_interpolation)

        return interpolation



    def interpolate_transfer_function(self, redshift :float=None, careful_interpolation :bool=False):
        '''lsReturns a transfer function interpolation object

        Args:
            redshift (float): Redshift to which the transfer function is evolved to
            careful_interpolation (bool): Parameter for careful extrapolation.

        Returns:
            An interpolation object
        '''
        if redshift is None:
            redshift = self.redshift
        self.parameters['output'] += 'mTk'
        self.model = Class()
        self.model.set(self.parameters)
        self.model.compute()
        transfer = self.model.get_transfer(redshift)
        alpha = interpolate.interp1d(transfer['k (h/Mpc)']*self.model.h()*self.factor, transfer['d_cdm'],
                fill_value='extrapolate', bounds_error=careful_interpolation)

        return alpha


    def fnl_halo_powerspectrum(self, fnl :float=0, bias :float=2.0, deltac :float=1.686, p:float=1.0)-> Box:
        """Returns the powerspectrum for a given value of fnl
        Keyword Arguments:
            fnl {int} -- The value of fnl to use (default: {0})
            bias {float} -- The bias of halos wrt matter (default: {2.0})
            deltac {float} -- Crtical collapse parameter (default: {1.686})
            p {float} -- (default: {1.0})
        Returns:
            Box -- A Box object containing k and Pk
        """

        k = numpy.geomspace(1e-4,1e2, 5000)
        coefficient = (bias + 2*deltac*(bias-p)*fnl/self.interpolate_transfer_function()(k))
        powerspectrum = coefficient**2 * self.interpolate_matter_powerspectrum()(k)

        return Box({'k': k, 'powerspectrum': powerspectrum})



    def cmb_noise_map(self, snapshot, white_noise: float=6e-6, beam_width: float=2, voxels: int=None, mode='real', bins: int=20)-> numpy.ndarray:
        arcmin_to_radian = numpy.pi/180./60.
        if voxels is None:
            voxels = min(int(snapshot.boxsize), 1024)

        gridding_factor = snapshot.boxsize/voxels
        chi_star = self.Chistar(snapshot.redshift)
        freq = 2*numpy.pi*numpy.fft.fftfreq(voxels, gridding_factor)
        kx, ky = numpy.meshgrid(freq, freq)
        lnorm = numpy.sqrt(kx**2 + ky**2)[:,:int(len(freq)/2 + 1)]*chi_star
        lnorm[0,0] = 1
        noise = (white_noise*arcmin_to_radian)**2 * numpy.exp((lnorm*(lnorm+1)*(arcmin_to_radian*beam_width)**2)/(8*numpy.log(2)))
        noise = noise*(snapshot.boxsize*chi_star)**2/2
        std2d = numpy.sqrt(noise)
        map2d = numpy.random.normal(scale=std2d) + 1j*numpy.random.normal(scale=std2d)

        if mode == 'real':
            return numpy.fft.irfft2(map2d).real
        else:
            return map2d



    def noiseck_from_simulation(self, snapshot, white_noise: float=6e-6, beam_width: float=2, voxels: int=None, bins=20)-> numpy.ndarray:

        if voxels is None:
            voxels = min(int(snapshot.boxsize), 1024)

        noise_map = self.cmb_noise_map(snapshot, white_noise, beam_width,  voxels=voxels, mode='complex')
        k, Ck = pypowerspectrum.powerspectrum2D(noise_map, boxsize=snapshot.boxsize, bins=bins)
        powerspectrum = Box({'k': k, 'Ck': Ck})
        return powerspectrum


    def Noisekmap(self, snapshot, pixels, pfile=None):

        chi_star = self.Chistar(snapshot.redshift)
        k2d = kineticsz.utils.k2D(snapshot.boxsize, pixels)
        try:
            noiseck = numpy.load(pfile)
        except:
            noiseck = numpy.load('/home/ugiri/projects/kineticsz/data/Ck_sw_0.5_beam_1_in_mpc_out_mpc_hr.npz')
        el = noiseck['k']*chi_star*self.factor
        Cl = (noiseck['k']**2/el**2)*(noiseck['noiseck'] + noiseck['cmbck'])
        Clcmb = Cl*(chi_star*snapshot.boxsize)**2
        std2D = numpy.sqrt(numpy.interp(k2d.flatten(), el/chi_star, Clcmb/2)).reshape(k2d.shape)
        map2D = numpy.random.normal(scale=std2D) + 1j*numpy.random.normal(scale=std2D)
        map2D[int(pixels/2 +1):, 0] = numpy.conj(map2D[1:int(pixels/2),0][::-1])
        return numpy.fft.irfft2(map2D)


