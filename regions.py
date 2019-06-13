import sys
import os.path
import numpy as np
from scipy.interpolate import interp1d

try:
    import h5py
except ImportError:
    raise ImportError('You don t have the package h5py installed. Try "pip install h5py".')

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError('You don t have the package tqdm installed. Try "pip install tqdm".')

try:
    import yt
except ImportError:
    raise ImportError('You don t have the package yt installed. Try "pip install yt".')

from yt import derived_field
from yt.units import Myr, pc

if len(sys.argv) > 4:
    raise RuntimeError('Too many arguments! Try "python3 regions.py "simulation" "region_radius_pc" "young_stars_age_Myr".')
elif len(sys.argv) < 4:
    raise RuntimeError('Not enough arguments! Try "python3 regions.py "simulation" "region_radius_pc" "young_stars_age_Myr.')


simulation = sys.argv[1]
name = 'G-' + str(sys.argv[1])[-4:]
region_radius_pc = float(sys.argv[2])
stars_age = float(sys.argv[3])
G = 4.30091e-3    # (pc/Msun)*(km/s)**2
pc_to_km = 3.086e+13
seconds_to_yr = 3.17098e-8


@yt.particle_filter(requires=["particle_type"], filtered_type='all')
def stars(pfilter, data):
    filter = data[(pfilter.filtered_type, "particle_type")] == 2
    return filter

@yt.particle_filter(requires=["age"], filtered_type='all')
def young_stars(pfilter, data):
    filter = data[(pfilter.filtered_type, "age")] < stars_age*Myr
    return filter

@derived_field(name="disk_z", units="pc", force_override=True)
def disk_z(field, data):
    return data["z"] - data.get_field_parameter('center')[2]


ds = yt.load(simulation + "/" + name)
ds.add_particle_filter('stars')
ds.add_particle_filter('young_stars')
ad = ds.all_data()
print('Simulation data succesfully loaded.')
print('Computing radius that encompasses 90% of mass...')
MassFraction = 0.9
Mass = ad[("young_stars", "particle_mass")]
x = ad[("young_stars", "particle_position_relative_x")].in_units("kpc")
y = ad[("young_stars", "particle_position_relative_y")].in_units("kpc")
Radius = np.sqrt(x**2 + y**2)
Mass = Mass[Radius.argsort()]
Mass = np.cumsum(Mass)
Mass /= Mass[-1]
Radius = Radius[Radius.argsort()]
f = interp1d(Mass, Radius)
radius = abs(f(MassFraction))


def positions_xy(r, R):
    pos = []
    t_max = int(R/(2*r))
    for t in range(-t_max, t_max+1):
        for q in range(-t_max, t_max+1):
            pos.append([2*r*t, 2*r*q])
    pos2 = []
    for point in pos:
        if np.sqrt(point[0]**2 + point[1]**2) + r < R:
            pos2.append(point)
    return pos2


def positions_zh(dataset, pos_xy, r):
    center = dataset.domain_center.in_units("pc")
    pos_xy = pos_xy*pc
    pos_z = []
    height = []
    for i in tqdm(range(len(pos_xy)), total=len(pos_xy)):
        center_xyz = np.concatenate((np.array(pos_xy[i]), np.array([0.0])))
        disk = dataset.disk(center_xyz*pc + center, [0, 0, 1], (r, "pc"), (2000.0, "pc"))
        variance, center_mass_z = disk.quantities.weighted_variance(("gas", "disk_z"), "density")
        pos_z.append(float(center_mass_z))
        height.append(3*float(variance))
    return pos_z, height


def positions_xyz(xy, z, h):
    pos = []
    for i in range(len(xy)):
        q = np.array([(xy[i])[0], (xy[i])[1], z[i], h[i]])
        pos.append(q)
    return pos


print("Computing regions' positions and dimensions...")
centers_xy = positions_xy(region_radius_pc, 1000*radius)
centers_z, heights = positions_zh(ds, centers_xy, float(region_radius_pc))
centers = np.array(positions_xyz(centers_xy, centers_z, heights))*pc
print("Regions created.")


print("Creating h5py file...")
if not os.path.exists("regions_data"):
    os.makedirs("regions_data")
filename = "regions_data/" + name + "_" + str(region_radius_pc) + "pc_" + str(stars_age) + "_Myr_position.h5"
file = h5py.File(filename, 'w')
regions = file.create_group('regions_data')
parameters = file.create_group('parameters')
regions.create_dataset('positions', data=centers)
parameters.create_dataset('radius90', data=[radius])
print(filename + " created, with groups 'parameters' and 'regions_data'.")
print("Regions' positions and dimensions, and 90%-mass radius saved in 'parameters' group.")


print("Obtaining yt fields data (this might take a while)...")
center = ds.domain_center.in_units("pc")
ad = ds.disk(center, [0, 0, 1], (radius*1000, "pc"), (2000.0, "pc"))
positions = centers
gas_x = ad[("gas", "x")].in_units("pc") - center[0]
gas_y = ad[("gas", "y")].in_units("pc") - center[1]
gas_z = ad[("gas", "z")].in_units("pc") - center[2]
gas_vx = ad[("gas", "velocity_x")].in_units("km/s")
gas_vy = ad[("gas", "velocity_y")].in_units("km/s")
gas_vz = ad[("gas", "velocity_z")].in_units("km/s")
gas_v_theta = ad[("gas", "velocity_cylindrical_theta")].in_units("km/s")
young_stars_x = ad[("young_stars", "relative_particle_position_x")].in_units("pc")
young_stars_y = ad[("young_stars", "relative_particle_position_y")].in_units("pc")
young_stars_z = ad[("young_stars", "relative_particle_position_z")].in_units("pc")
stars_x = ad[("stars", "relative_particle_position_x")].in_units("pc")
stars_y = ad[("stars", "relative_particle_position_y")].in_units("pc")
stars_z = ad[("stars", "relative_particle_position_z")].in_units("pc")
gas_mass = ad[("gas", "cell_mass")].in_units("Msun")
gas_density = ad[("gas", "density")].in_units("Msun/pc**3")
gas_number_density = ad[("gas", "number_density")].in_units("1/cm**3")
gas_sound_speed = ad[("gas", "sound_speed")].in_units("km/s")
young_stars_mass = ad[("young_stars", "particle_mass")].in_units("Msun")
stars_mass = ad[("stars", "particle_mass")].in_units("Msun")
dense_gas_mass = gas_mass[gas_number_density > 100.0]      # Msun
dense_gas_x = gas_x[gas_number_density > 100.0]     # pc
dense_gas_y = gas_y[gas_number_density > 100.0]     # pc
dense_gas_z = gas_z[gas_number_density > 100.0]     # pc
region_x = positions[:, 0]
region_y = positions[:, 1]
region_z = positions[:, 2]
region_H = positions[:, 3]


print('Computing regions gas mass (Msun)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = gas_mass[((gas_x - region_x[i]) ** 2 + (gas_y - region_y[i]) ** 2 < region_radius_pc ** 2) &
                    (abs(gas_z - region_z[i]) < region_H[i] / 2.0)].sum()
    field.append(data)
regions.create_dataset('gas_mass', data=field)
regions_gas_mass = field

print('Computing regions young stars mass (Msun)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = young_stars_mass[((young_stars_x - region_x[i])**2 + (young_stars_y - region_y[i])**2 < region_radius_pc**2)
                            & (abs(young_stars_z - region_z[i]) < region_H[i]/2.0)].sum()
    field.append(data)
regions.create_dataset('young_stars_mass', data=field)
regions_young_stars_mass = field

print('Computing regions stars mass (Msun)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = stars_mass[((stars_x - region_x[i])**2 + (stars_y - region_y[i])**2 < region_radius_pc**2)
                        & (abs(stars_z - region_z[i]) < region_H[i]/2.0)].sum()
    field.append(data)
regions.create_dataset('stars_mass', data=field)
regions_stars_mass = field

print('Computing dense gas mass (Msun)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = dense_gas_mass[((dense_gas_x - region_x[i])**2 + (dense_gas_y - region_y[i])**2 < region_radius_pc**2)
                          & (abs(dense_gas_z - region_z[i]) < region_H[i]/2.0)].sum()
    field.append(data)
regions.create_dataset('dense_gas_mass', data=field)

print('Computing regions galactic radius (pc)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = np.sqrt(region_x[i]**2 + region_y[i]**2)
    field.append(data)
regions.create_dataset('galactic_radius', data=field)
galactic_radius = field

print('Computing regions x-velocity dispersion (km/s)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    mean = np.mean(gas_vx[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                          (abs(gas_z - region_z[i]) < region_H[i]/2.0)])
    mean_squares = np.mean(gas_vx[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                           (abs(gas_z - region_z[i]) < region_H[i]/2.0)]**2)
    data = np.sqrt(mean_squares - mean**2)
    field.append(data)
regions.create_dataset('velocity_dispersion_x', data=field)
dispersion_vx = field

print('Computing regions y-velocity dispersion (km/s)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    mean = np.mean(gas_vy[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                          (abs(gas_z - region_z[i]) < region_H[i]/2.0)])
    mean_squares = np.mean(gas_vy[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                           (abs(gas_z - region_z[i]) < region_H[i]/2.0)]**2)
    data = np.sqrt(mean_squares - mean**2)
    field.append(data)
regions.create_dataset('velocity_dispersion_y', data=field)
dispersion_vy = field

print('Computing regions z-velocity dispersion (km/s)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    mean = np.mean(gas_vz[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                          (abs(gas_z - region_z[i]) < region_H[i]/2.0)])
    mean_squares = np.mean(gas_vz[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                           (abs(gas_z - region_z[i]) < region_H[i]/2.0)]**2)
    data = np.sqrt(mean_squares - mean**2)
    field.append(data)
regions.create_dataset('velocity_dispersion_z', data=field)
dispersion_vz = field

print('Computing regions 3d-velocity dispersion (km/s)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = np.sqrt(dispersion_vx[i]**2 + dispersion_vy[i]**2 + dispersion_vz[i]**2)
    field.append(data)
regions.create_dataset('velocity_dispersion_3d', data=field)
dispersion_3d = field

print('Computing regions mach number 3d...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = dispersion_3d[i] / \
           np.median(gas_sound_speed[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                                     (abs(gas_z - region_z[i]) < region_H[i]/2.0)])
    field.append(data)
regions.create_dataset('mach_number_3d', data=field)

print('Computing regions mach number z...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = dispersion_vz[i] / \
           np.median(gas_sound_speed[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                                     (abs(gas_z - region_z[i]) < region_H[i]/2.0)])
    field.append(data)
regions.create_dataset('mach_number_z', data=field)

print('Computing regions virial parameter 3d...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = 5.0*dispersion_3d[i]**2 * region_radius_pc/(regions_gas_mass[i]*G)
    field.append(data)
regions.create_dataset('virial_parameter_3d', data=field)

print('Computing regions virial parameter z...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = 5.0*dispersion_vz[i]**2 * region_radius_pc/(regions_gas_mass[i]*G)
    field.append(data)
regions.create_dataset('virial_parameter_z', data=field)

print('Computing regions mean density (Msun/pc**3)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = regions_gas_mass[i] / (np.pi*region_radius_pc**2*region_H[i])
    field.append(data)
regions.create_dataset('mean_density', data=field)
regions_mean_density = field

print('Computing regions Sigma Gas (Msun/pc**2)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = regions_gas_mass[i] / (np.pi * region_radius_pc ** 2)
    field.append(data)
regions.create_dataset('sigma_gas', data=field)
sigma_gas = field

print('Computing regions Sigma Stars (Msun/pc**2)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = regions_stars_mass[i] / (np.pi * region_radius_pc ** 2)
    field.append(data)
regions.create_dataset('sigma_stars', data=field)

print('Computing regions Sigma SFR (Msun/yr*kpc**2)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = regions_young_stars_mass[i]/(stars_age*np.pi*region_radius_pc**2)
    field.append(data)
regions.create_dataset('sigma_SFR', data=field)
sigma_SFR = field

print('Computing regions angular velocity (1/Myr)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    if galactic_radius[i] == 0.0:
        data = 0.0
    else:
        data = (np.mean(gas_v_theta[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 < region_radius_pc**2) &
                                    (abs(gas_z - region_z[i]) < region_H[i]/2.0)])/pc_to_km)/galactic_radius[i]
    field.append(data*1.0e6/seconds_to_yr)
regions.create_dataset('angular_velocity', data=field)
angular_velocity = field

print('Computing regions crossing time (Myr)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = (region_H*pc_to_km /
            np.sqrt(dispersion_3d[i]**2 +
                    np.median(gas_sound_speed[((gas_x - region_x[i])**2 + (gas_y - region_y[i])**2 <
                                               region_radius_pc**2) & (abs(gas_z - region_z[i]) <
                                                                       region_H[i] / 2.0)])**2))
    field.append((seconds_to_yr/1000000.0)*data)
regions.create_dataset('crossing_time', data=field)

print('Computing regions free fall time (Myr)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    data = (3.0*np.pi/np.sqrt(32.0*G*(regions_mean_density[i] / (pc_to_km**2)))) * (seconds_to_yr/1.0e6)
    field.append(data)
regions.create_dataset('free_fall_time', data=field)
free_fall_time = field

print('Computing regions depletion time (Myr)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    if sigma_SFR[i] == 0.0:
        data = 0.0
    else:
        data = (sigma_gas[i]/sigma_SFR[i])
    field.append(data)
regions.create_dataset('depletion_time', data=field)
depletion_time = field

print('Computing regions orbital time (2*pi/Omega)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    if angular_velocity[i] == 0.0:
        data = 0.0
    else:
        data = (2.0*np.pi/angular_velocity[i])
    field.append(data)
regions.create_dataset('orbital_time', data=field)

print('Computing regions SFR efficiency (t_ff/t_dep)...')
field = []
for i in tqdm(range(len(positions)), total=len(positions)):
    if depletion_time[i] == 0.0:
        data = 0.0
    else:
        data = (free_fall_time[i]/depletion_time[i])
    field.append(data)
regions.create_dataset('SFR_efficiency', data=field)

file.close()
print("Regions' data succesfully saved in 'regions_data' group.")
