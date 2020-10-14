import numpy

def bootstrap(array, function, percentile=95, num=1000):
    estimates = [function(numpy.random.choice(array, size=len(array), replace=True)) for _ in range(num)]
    return tuple(numpy.percentile(estimates, [(100-percentile)/2, (100+percentile)/2]))
