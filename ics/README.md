To run:

```python zeldovich.py --arguments```

<pre>arguments:
  --boxsize         size of simulation box in the desired output units 
  --pixels          number of grids used 
  --redshift        redshift at which initial conditions are desired
  --read_hdf5       A hdf5 file that is copied along with its attributes
  --write_hdf5      the output hdf5 file containing the initial data to be fed to Gadget
  --fnl             fnl value to be used
  --seed            seed for random number generator
  --out_units       unit in which particle position are written could be `kpc`, `kpch`, `mpc` or `mpch`
  --scaling         type of transfer function scaling to be used either `trivial` or `non-trivial`
  --class_config    json file containg configuration for CLASS
  
</pre>
