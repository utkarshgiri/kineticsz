import h5py
import fire
import numpy
import pathlib
import getdist
import utensils
import kineticsz
import matplotlib
from matplotlib import pyplot as plt
from getdist import plots, MCSamples

colormap = matplotlib.colors.ListedColormap(utensils.palletes.plotly_color)

def plotter(lastn=500, form='1d', fnl=0, basename=None, *filenames):
    """Usage:
        ```python corner_plots.py --lastn=50000 --form=2d --basename=combined_minus50 --fnl=-50 ../../mcmc_chains/combined_velocity_quijote_plus50.h5```
    """
    files = sorted(filenames)
  
    def mc_samples(filename, plot=True, label=None):
        stds = []
        try:
            samples = h5py.File(filename, 'r')['mcmc']['chain'][:]
            parameters = samples.shape[-1]
            
            if parameters == 2:
                if form == '1d':
                    axes = [0,1]; labels = ['{b_g}', '{f_{NL}}']
                elif form == '2d':
                    dsamples = numpy.ones(shape=(samples.shape[0], samples.shape[1]+1))
                    dsamples[:,[0,-1]] = samples
                    samples = dsamples
                    axes = [0,1,2]; labels = ['{b_g}', '{b_v}',  '{f_{NL}}']
            
            if parameters == 3:
                if form == '1d':
                    axes = [0,2]; labels = ['{b_g}', '{f_{NL}}']
                else:
                    axes = [0,1,2]; labels = ['{b_g}', '{b_v}',  '{f_{NL}}']
            
            median, std = numpy.median(samples, axis=0), numpy.std(samples, axis=0)
            stds.append(std)

            if plot:
                samples = getdist.MCSamples(samples=samples.reshape(-1,parameters)[:][-lastn:,axes],
                                        names=labels,
                                        labels=labels,
                                        label=label,
                                        settings={'fine_bins_2D':32, 'smooth_scale_1D':0.3})
                return samples

        except Exception as e:
            print(e)
    
    stds =[numpy.std(h5py.File(filename, 'r')['mcmc']['chain'][-10:,:,-1]) for filename in filenames]
    filenames = [x for _,x in sorted(zip(stds, filenames))]
    labelnames = []
    if 'density' in ''.join(filenames): labelnames.append('[$\delta_h, \delta_m$]')
    if 'velocity' in ''.join(filenames): labelnames.append('[$\delta_h, v_r$]')
    if 'halo' in ''.join(filenames): labelnames.append('[$\delta_h$]')
    labels = ['{b_g}', '{b_v}',  '{f_{NL}}']  
    
    samples = [mc_samples(name, label=label) for (name, label) in zip(filenames, labelnames)]
    contour_ls = [(0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1))] 

    markers = {key: value for (key, value) in zip(labels[1:], [1, fnl])}
    #g = plots.get_subplot_plotter()
    #g.settings.solid_colors = colormap
    #g.settings.alpha_filled_add = 1.0
    #g.settings.figure_legend_frame = False
    #markers = {key: value for (key, value) in zip(labels[1:], [1,0])}
    g = plots.get_subplot_plotter(width_inch=5)
    #g.settings.solid_colors = 'tab10'
    g.settings.alpha_filled_add = 1
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = 15
    g.settings.axes_fontsize=15
    g.settings.fontsize = 18
    g.settings.axes_labelsize = 15
 
    #g.triangle_plot(list(reversed(samples)), filled=True, title_limit=1, contour_lws=1.2, contour_ls=contour_ls, contour_colors=['b', 'r', 'k'])
    #g.triangle_plot(list(reversed(samples)), filled=True, title_limit=1, contour_lws=1.2, contour_colors=['b', 'r', 'k'])
    g.triangle_plot(list(reversed(samples)), filled=True, title_limit=1, contour_lws=1.2)


    import uuid
    plotname = basename if basename is not None else str(uuid.uuid4())
    plotname = '../plots/mcmc_{}.pdf'.format(plotname)
    g.export(plotname)

    try:
        utensils.save_to_mac.save(filename=plotname, dpi=500)
    except:
        utensils.save_and_upload_plot(filename=plotname, dpi=500, subdirectory='ksz_mcmc')


if __name__ == '__main__':
    fire.Fire(plotter)
