import numpy
import joblib
import pathlib
import kineticsz
from scipy.optimize import curve_fit

cmb = kineticsz.pycmb.CMB(parameters='/home/ugiri/kineticsz/zeldovich/configuration/class_quijote.json')

alpha = cmb.interpolate_transfer_function(redshift=2, careful_interpolation=False)

deltac = 1.42
def model(k, bg, fnl):
    return bg + 2*deltac*(bg-1)*fnl/alpha(k)

dirnames = list(pathlib.Path('/gpfs/ugiri/gadget/size1024/').glob('fnlplus50z127n[45]')) 
dirnames +=  list(pathlib.Path('/gpfs/ugiri/gadget/size1024/').glob('fnl50z127n[2]'))
print(dirnames)

bias = []
for name in dirnames:
    h = numpy.load(name.joinpath('flattened_modes.npz'))
    k = h['k']
    pmh = h['hm'][k<0.012]*h['dm'][k<0.012].conjugate()
    pmm = h['dm'][k<0.012]*h['dm'][k<0.012].conjugate()
    bias.append(pmh/pmm)
k = k[k<0.012]

bias = numpy.mean(bias, axis=0)

parameters, covariance = curve_fit(model, k, bias.real)

print('bg, fnl = ({}, {})'.format(parameters[0], parameters[1]))
print('Error on bg, fnl = ({}, {})'.format(numpy.sqrt(covariance[0,0]), numpy.sqrt(covariance[1,1,])))

