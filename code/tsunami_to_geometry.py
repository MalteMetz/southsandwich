import re
import os
import os.path as op

import numpy as num
from scipy.interpolate import interp2d

from pyrocko.model import geometry
from pyrocko import table, util


d2r = num.pi / 180.
km = 1e3

path_results = ('/home/malde/Documents/Jobs_Praktika/PhD/publications/'
                'southsandwich/tsunami/data')

pattern = r'eWave\.(?P<hr>[0-9]{2})h(?P<mn>[0-9]{2})\.ssh.ef'

fns = os.listdir(path_results)
matches = [re.match(pattern, fn) for fn in fns if re.match(pattern, fn)]

times = num.array([
    float(m.group('hr')) * 3600. + float(m.group('mn')) * 60.
    for m in matches])

idx = num.argsort(times)[::-1]
times_sorted = times[idx]
matches_sorted = [matches[i] for i in idx]

n_times = times_sorted.shape[0]


dx = .25  # Cell size in [deg]

data = []
for i, m in enumerate(matches_sorted):
    fn = op.join(path_results, m.string)

    print(fn)

    with open(fn, 'r') as f:
        head = [next(f) for x in range(6)]

    ncols = int(head[0].split()[1])  # extend in lon
    nrows = int(head[1].split()[1])  # extend in lat
    xllcenter = float(head[2].split()[1])  # lon of lower left cell center
    yllcenter = float(head[3].split()[1])  # lat of lower left cell center
    cellsize = float(head[4].split()[1])
    nodata_v = int(head[5].split()[1])

    fn_data = num.loadtxt(fn, skiprows=6)

    xllcenter = num.around(xllcenter, decimals=4)
    yllcenter = num.around(yllcenter, decimals=4)
    cellsize = num.around(cellsize, decimals=4)

    lat = num.linspace(yllcenter, yllcenter + (nrows - 1) * cellsize, nrows)
    lon = num.linspace(xllcenter, xllcenter + (ncols - 1) * cellsize, ncols)

    f = interp2d(lon, lat, fn_data[::-1, :], kind='cubic')

    if i == 0:
        lat_interp = num.arange(lat.min(), lat.max() + dx, dx)
        lon_interp = num.arange(lon.min(), lon.max() + dx, dx)

    data_interp = f(lon_interp, lat_interp)
    # data_interp[num.abs(data_interp < 0.05)] = num.nan

    data.append(data_interp.flatten())

# Static wave height data
fn = op.join(path_results, 'eWave.ewh.ef')
with open(fn, 'r') as f:
    head = [next(f) for x in range(6)]

ncols = int(head[0].split()[1])  # extend in lon
nrows = int(head[1].split()[1])  # extend in lat
xllcenter = float(head[2].split()[1])  # lon of lower left cell center
yllcenter = float(head[3].split()[1])  # lat of lower left cell center
cellsize = float(head[4].split()[1])
nodata_v = int(head[5].split()[1])

fn_data = num.loadtxt(fn, skiprows=6)

xllcenter = num.around(xllcenter, decimals=4)
yllcenter = num.around(yllcenter, decimals=4)
cellsize = num.around(cellsize, decimals=4)

lat = num.linspace(yllcenter, yllcenter + (nrows - 1) * cellsize, nrows)
lon = num.linspace(xllcenter, xllcenter + (ncols - 1) * cellsize, ncols)

f = interp2d(lon, lat, fn_data[::-1, :], kind='cubic')

wave_height = f(lon_interp, lat_interp)
# wave_height[num.abs(wave_height < 0.1)] = num.nan


data = data[::-1]
times_sorted = times_sorted[::-1]

assert len(set([d.shape for d in data])) == 1

# Vertices
n_lat = lat_interp.shape[0]
n_lon = lon_interp.shape[0]
n_points = n_lat * n_lon

vertices = num.zeros((n_points * 4, 5))
vertices[:, 0] = num.repeat(num.repeat(lat_interp, 4), n_lon)
vertices[:, 1] = num.tile(num.repeat(lon_interp, 4), n_lat)

vertices[::4, 0] += dx / 2.
vertices[1::4, 0] -= dx / 2.
vertices[2::4, 0] -= dx / 2.
vertices[3::4, 0] += dx / 2.

vertices[::4, 1] += dx / 2.
vertices[1::4, 1] += dx / 2.
vertices[2::4, 1] -= dx / 2.
vertices[3::4, 1] -= dx / 2.

faces = num.zeros((n_points, 5), dtype=num.int64)
faces[:, :-1] = num.array([
    num.arange(ip, ip+4) for ip in range(0, n_points * 4, 4)])
faces[:, -1] = faces[:, 0]

times_sorted += util.stt('2021-08-12 18:34:46')
print([util.tts(t) for t in times_sorted])

geom = geometry.Geometry()
geom.set_vertices(vertices)
geom.set_faces(faces)

geom.times = times_sorted

geom.add_property(
    name=table.Header(
        name='height [m]',
        label='height [m]',
        sub_headers=['']),
    values=wave_height.flatten())

geom.add_property(
    name=table.Header(
        name='amp [m]',
        label='amp [m]',
        sub_headers=['' for i in range(times_sorted.shape[0])]),
    values=num.vstack([d for d in data]).T)

geom.dump(filename='../sparrow/tsunami_geom.yaml')
