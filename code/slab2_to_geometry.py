import re
import os
import os.path as op

import numpy as num
from scipy.interpolate import RegularGridInterpolator, interp1d

import shapefile  # PyShp package

import matplotlib.pyplot as plt

from pyrocko import orthodrome as pod
from pyrocko.model import geometry
from pyrocko import table


d2r = num.pi / 180.
km = 1e3

fault_decimation = 1.  # Decimates number of grid points to estimate subfaults
depth_min = 0.  # Minimum depth of plane top edge in m
patch_size = 5000.  # Fault patch size in m

path_slab2 = '/home/malde/Documents/Jobs_Praktika/PhD/databases/slab2'
rake1 = 0.
rake2 = 0.


fault_path = op.join(path_slab2, 'sco_shapefiles/sco_shapefiles')
f = 'sco_depth'

kwargs = dict([
    (ext, open(os.path.join(fault_path, '.'.join([f, ext])), 'rb'))
    for ext in ['shp', 'prj', 'dbf', 'shx', 'sbn']])

shape = shapefile.Reader(encoding='ISO-8859-1', **kwargs)
geoj = shape.shapeRecord(0).__geo_interface__

contours = {}
for sh in shape.shapeRecords():
    geoj = sh.__geo_interface__

    depth = geoj['properties']['DEPTH']

    if depth > 260.:
        continue

    if depth in contours:
        raise ValueError('depth %f already in contours', depth)

    lats, lons = [], []

    for pair in geoj['geometry']['coordinates']:
        lats.append(pair[1])
        lons.append(pair[0])

    lats = num.array(lats)
    lons = num.array(lons) % 360.
    lons[lons > 180.] -= 360.

    contours[depth] = dict(lats=lats, lons=lons)

lon_clp, lat_clp = num.loadtxt(
    op.join(
        path_slab2,
        'Slab2Distribute_Mar2018/Slab2Clips/sco_slab2_clp_02.23.18.csv'),
    unpack=True)
lon_clp[lon_clp > 180.] -= 360.


# Make geometry
n_points = len(contours[20]['lats'])

geom = geometry.Geometry()

vertices = []
depths = list(contours.keys())
depths.sort()

for d in depths:
    c = contours[d]
    lats = c['lats']
    lons = c['lons']

    azis, dists = pod.azidist_numpy(
        num.array([lats[0]]), num.array([lons[0]]),
        num.array(lats), num.array(lons))

    dists = num.around(dists, decimals=5)

    dists_interp = num.linspace(0, dists.max(), n_points)

    f = interp1d(dists, azis)
    azis_interp = f(dists_interp)

    lats, lons = pod.azidist_to_latlon(
        lats[0], lons[0],
        azis_interp, dists_interp)

    vertices += [[lat, lon, 0., 0., d*1e3] for lat, lon in zip(lats, lons)]

    contours[d]['lats'] = lats
    contours[d]['lons'] = lons

vertices = num.array(vertices)
geom.set_vertices(vertices)

faces = []
idx_min = 0

for d_top, d_btm in zip(depths[:-1], depths[1:]):
    len_tot = n_points * 2

    fcs = num.zeros((n_points - 1, 5), dtype=num.int64)
    fcs[:, :] = num.array([
        [ip, ip + 1, ip + n_points + 1, ip + n_points, ip]
        for ip in num.arange(idx_min, idx_min + n_points - 1)])

    faces.append(fcs)

    idx_min += n_points

faces = num.vstack(faces)
geom.set_faces(faces)

geom.times = num.array([0.])

geom.add_property(
    name=table.Header(
        name='none',
        label='none',
        sub_headers=['']),
    values=num.ones(faces.shape[0]))

print(geom.T.propnames)
print(geom.properties.description)

geom.dump(filename='../sparrow/slab2_geom.yaml')
