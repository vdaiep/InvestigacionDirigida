# InvestigacionDirigida
Enzo galaxy simulations analysis.

The script regions.py loads an Enzo galaxy simulation, and divides the data into smaller cylindrical regions. Each region is considered as a molecular cloud, and several parameters are computed.

Run the script on a terminal as:
```
python3 regions.py "simulation" "radius" "age"
```
where "simulation" the directory where the simulation data is, "radius" is the radius (pc) of each region and "age" is the age (Myr) of the stars considered young to calculate the stellar formation rate (usually 10 Myr). The output is a HDF5 file, which contains the following data of each region on the group "regions_data":

- Gas mass (Msun)
- Young stars mass (Msun)
- Stars mass (Msun)
- Dense gas mass(Msun) (Dense if gas number density > 1 cm^-3)
- Galactocentric radius (pc)
- x-velocity dispersion (km/h)
- y-velocity dispersion (km/h)
- z-velocity dispersion (km/h)
- 3d-velocity dispersion (km/h)
- Mach number z
- Mach number 3d
- Virial parameter z
- Virial parameter 3d
- Mean density (Msun/pc^3)
- Gas surface density (Msun/pc^2)
- Stars surface density (Msun/pc^2)
- SFR surface density (Msun/yr kpc^2)
- Angular velocity (1/Myr)
- Crossing time (Myr)
- Free-fall time (Myr)
- Depletion time (Myr)
- Orbital time (Myr)
- SFR efficiency (t_ff/t_dep)


