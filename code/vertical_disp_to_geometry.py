import os.path as op

import numpy as num

from pyrocko.model import geometry
from pyrocko import table, util


d2r = num.pi / 180.
km = 1e3

path_results = ('/home/malde/Documents/Jobs_Praktika/PhD/publications/'
                'southsandwich/tsunami')

lat, lon, un, ue, ud = num.loadtxt(
    op.join(path_results, 'displacement_pdr_wphase.csv'),
    unpack=True,
    skiprows=1)

n_points = lat.shape[0]

vertices = num.zeros((n_points * 4, 5))
vertices[:, 0] = num.repeat(lat, 4)
vertices[:, 1] = num.repeat(lon, 4)

for ip in range(n_points):
    vertices[ip*4:(ip+1)*4, 2:4] = num.array([
        [2e3, 2e3],
        [2e3, -2e3],
        [-2e3, -2e3],
        [-2e3, 2e3]])

faces = num.zeros((n_points, 5), dtype=num.int64)
faces[:, :-1] = num.array([
    num.arange(ip, ip+4) for ip in range(0, n_points * 4, 4)])
faces[:, -1] = faces[:, 0]

time = num.array([util.stt('2021-08-12 18:34:46')])

geom = geometry.Geometry()
geom.set_vertices(vertices)
geom.set_faces(faces)

geom.times = time

geom.add_property(
    name=table.Header(
        name='uplift',
        label='uplift',
        sub_headers=['']),
    values=-ud)

geom.dump(filename='../sparrow/vertical_displacement_geom.yaml')
