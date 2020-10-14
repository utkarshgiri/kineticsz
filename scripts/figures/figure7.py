import h5py
import fire
import numpy
import getdist
import utensils
import kineticsz
import matplotlib
from matplotlib import pyplot as plt
from getdist import plots, MCSamples
plt.style.use(['science'])
colormap = matplotlib.colors.ListedColormap(utensils.palletes.plotly_color)
#colormap = matplotlib.colors.ListedColormap(['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])

def plotter(basename, fnl=0, form='1d', boxsize=2048, lastn=None, dirnum=''):

    
    if form == '2d':
        axes = [0,1,2]; labels = ['{b_g}', '{b_v}',  '{f_{nl}}']
    elif form == '1d':
        axes = [0,2]; labels = ['{b_g}', '{f_{nl}}']
    else:
        raise Exception('form is 1d or 2d')
    hsamples = h5py.File('/home/ugiri/kineticsz/mcmc_chains/combined_halo_quijote_58.h5', 'r')['mcmc']['chain'][:].reshape(-1,2)[:]
    samples = hsamples
    #samples = numpy.random.normal(size=(hsamples.shape[0], hsamples.shape[1]+1), loc=1, scale=1e-10)
    #samples = numpy.ones((hsamples.shape[0], hsamples.shape[1]+1))
    #samples[:,[0,-1]] = hsamples
    hmedian, hstd = numpy.median(samples, axis=0), numpy.std(samples, axis=0)
    print('Median from halos ', hmedian)
    hsamples = getdist.MCSamples(samples=samples[-lastn:,:],
                     names=labels,
                     labels=labels,
                     label='[$\delta_h$]',
                     settings={'fine_bins_2D':32, 'smooth_scale_1D':0.3})
    #msamples = h5py.File('../data/a{}z2.0_{}_{}momentum_samples.h5'.format(basename, dirnum, boxsize), 'r')['mcmc']['chain'][:].reshape(-1,3)[:]
    '''
    msamples = h5py.File('../data/combined_momentum_fnl50.h5', 'r')['mcmc']['chain'][:].reshape(-1,3)[30000:]
    msamples = msamples[:,axes]
    mmedian, mstd = numpy.median(msamples, axis=0), numpy.std(msamples, axis=0)
    print('Median from momentum ', mmedian)
    
    msamples = getdist.MCSamples(samples=msamples[-lastn:,:],
                     names=labels,
                     labels=labels,
                     label='[$\delta_h, q_k$]',
                     settings={'fine_bins_2D':64})
    '''
    #vsamples = h5py.File('../data/a{}z2.0_{}_{}velocity_samples.h5'.format(basename, dirnum, boxsize),'r')['mcmc']['chain'][:].reshape(-1,3)[:]
    vsamples = h5py.File('/home/ugiri/kineticsz/mcmc_chains/combined_velocity_quijote_58.h5', 'r')['mcmc']['chain'][:].reshape(-1,3)[:,:]
    vmedian, vstd = numpy.median(vsamples, axis=0), numpy.std(vsamples, axis=0)
    print('Median from velocity ', vmedian)

    vsamples = getdist.MCSamples(samples=vsamples[-lastn:,axes],
                     names=labels,
                     labels=labels,
                     label='[$\delta_h, \hat{v}_r$]',
                     settings={'fine_bins_2D':32, 'smooth_scale_1D':0.3})
    qsamples = h5py.File('/gpfs/ugiri/data_kineticsz/combined_velocity_quijote_58_true_noise.h5', 'r')['mcmc']['chain'][:].reshape(-1,3)[:,:]
    qmedian, vstd = numpy.median(qsamples, axis=0), numpy.std(qsamples, axis=0)
    print('Median from velocity ', qmedian)

    qsamples = getdist.MCSamples(samples=qsamples[-lastn:,axes],
                     names=labels,
                     labels=labels,
                     #label='[$\delta_h, \hat{v}_r$]',
                     settings={'fine_bins_2D':32, 'smooth_scale_1D':0.3})
 
    
    #dsamples = h5py.File('../data/a{}z2.0_{}_{}density_samples.h5'.format(basename, dirnum, boxsize), 'r')['mcmc']['chain'][:].reshape(-1,3)[:]
    dsamples = h5py.File('/home/ugiri/kineticsz/mcmc_chains/combined_density_quijote_58_modes.h5', 'r')['mcmc']['chain'][:].reshape(-1,2)[:,:]   
    #dsamples = dsamples[:,axes]
    samples = dsamples
    #samples = numpy.ones(shape=(dsamples.shape[0], dsamples.shape[1]+1))
    #samples[:,[0,-1]] = dsamples    
    dmedian, dstd = numpy.median(samples[-lastn:,:], axis=0), numpy.std(dsamples[-lastn:,:], axis=0)
    print('Median from density ', dmedian)

    dsamples = getdist.MCSamples(samples=samples[-lastn:,:],
                     names=labels,
                     labels=labels,
                     label='[$\delta_h, \delta_m$]',
                     settings={'fine_bins_2D':32, 'smooth_scale_1D':0.3})
    print(labels)
    markers = {key: value for (key, value) in zip(labels[1:], [0])}
    g = plots.get_subplot_plotter(width_inch=5)
    #g.settings.solid_colors = 'tab10'
    g.settings.alpha_filled_add = 1
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = 15
    g.settings.axes_fontsize=15
    g.settings.fontsize = 18
    g.settings.axes_labelsize = 15
    #lower_kwargs = {'contour_colors': cm.tab10.colors[4:], 'contour_ls': ['-', '--']}
    lower_kwargs = {'contour_ls': ['-', '-.', '--']}
    if form == '1d':
        #g.triangle_plot([hsamples, vsamples, dsamples], filled=False, contour_lws=1.2, line_args=[{'ls':'-', 'color':'k'},
        #    {'ls':'-.', 'color': 'r'}, {'ls': '--', 'color': 'b'}])

        #g.triangle_plot([hsamples, vsamples, dsamples], filled=True, contour_lws=1.2)
        g.triangle_plot(list(([hsamples, vsamples, dsamples])), contour_colors=['black', 'red', 'blue'], filled=True, contour_lws=1.2, markers=markers)# contour_ls=[(0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1))])

    elif form == '2d':
        #g.triangle_plot([vsamples, dsamples, hsamples], filled=True, title_limit=1, contour_lws=1.2, line_args=[{'ls':'-', 'color':'k'},
        #    {'ls':'-.', 'color': 'r'}, {'ls': '--', 'color': 'b'}])#, contour_ls=['-', '-.', '--'])
        #g.triangle_plot([hsamples, vsamples, dsamples], filled=True, contour_lws=1.2, line_args=[{'ls':':'},
        #    {'ls':'-.'}, {'ls': '--'}])#, contour_ls=['-', '-.', '--'])
        #g.triangle_plot(list(([vsamples, dsamples, hsamples])), filled=False, contour_lws=1.2, contour_ls=[(0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1))])
        g.triangle_plot(list(([hsamples, vsamples, dsamples])), filled=True, contour_lws=1.2)#, contour_ls=[(0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1))])


    plotname = 'figures/mcmc_{}_{}.pdf'.format(dirnum, fnl)
    plotnamepng = 'figures/mcmc_{}_{}.png'.format(dirnum, fnl)
    g.export(plotname)

    try:
        utensils.save_to_mac.save(filename=plotname, dpi=500)
    except:
        utensils.save_and_upload_plot(filename=plotname, dpi=500, subdirectory='ksz_mcmc')
        utensils.save_and_upload_plot(filename=plotnamepng, dpi=500, subdirectory='ksz_mcmc')


if __name__ == '__main__':
    fire.Fire(plotter)
