import re
import os
import os.path as op

import numpy as num
from scipy.interpolate import RegularGridInterpolator, interp1d

import matplotlib.pyplot as plt

from pyrocko import orthodrome as pod, gf, moment_tensor as pmt
from pyrocko.model import geometry
from pyrocko import table


d2r = num.pi / 180.


engine = gf.get_engine()
store = engine.get_store('global_2s_v2')

# load sources
path_results = ('/home/malde/Documents/Jobs_Praktika/PhD/publications/'
                'southsandwich/solutions')

pdr_load = gf.PseudoDynamicRupture.load

sourceA = pdr_load(filename=op.join(path_results, 'event1_pdr_mean.yaml'))
sourceD = pdr_load(filename=op.join(path_results, 'event2_pdr_mean.yaml'))
sourceB = pdr_load(filename=op.join(path_results, 'doublepdr_mean0.yaml'))
sourceC = pdr_load(filename=op.join(path_results, 'doublepdr_mean1.yaml'))

sources = [sourceA, sourceB, sourceC, sourceD]

for src in sources:
    src.discretize_patches(store)


def patch_outline(patch, cs):
    x = num.array([-1., 1., 1., -1., -1.])
    y = num.array([-1., -1., 1., 1., -1.])

    ln, wd = patch.length, patch.width
    strike, dip = patch.strike, patch.dip

    def array_check(variable):
        if not isinstance(variable, num.ndarray):
            return num.array(variable)
        else:
            return variable

    x, y = array_check(x), array_check(y)

    if x.shape[0] != y.shape[0]:
        raise ValueError('Shapes of x and y mismatch')

    x, y = x * 0.5 * ln, y * 0.5 * wd

    points = num.hstack((
        x.reshape(-1, 1), y.reshape(-1, 1), num.zeros((x.shape[0], 1))))

    anch_x, anch_y = gf.seismosizer.map_anchor['center']
    points[:, 0] -= anch_x * 0.5 * ln
    points[:, 1] -= anch_y * 0.5 * wd

    rotmat = num.asarray(
        pmt.euler_to_matrix(dip * d2r, strike * d2r, 0.0))

    points_rot = num.dot(rotmat.T, points.T).T

    points_rot[:, 0] += patch.north_shift
    points_rot[:, 1] += patch.east_shift
    points_rot[:, 2] += patch.depth

    if cs == 'xyz':
        return points_rot
    elif cs == 'xy':
        return points_rot[:, :2]
    elif cs in ('latlon', 'lonlat', 'latlondepth'):
        latlon = pod.ne_to_latlon(
            patch.lat, patch.lon, points_rot[:, 0], points_rot[:, 1])
        latlon = num.array(latlon).T
        if cs == 'latlon':
            return latlon
        elif cs == 'lonlat':
            return latlon[:, ::-1]
        else:
            return num.concatenate(
                (latlon, points_rot[:, 2].reshape((len(points_rot), 1))),
                axis=1)


def pdr_to_geometry(source):
    n_patches = len(source.patches)

    vertices = num.zeros((n_patches * 4, 5))
    vertices[:, 0] = num.repeat(source.get_patch_attribute('lat'), 4)
    vertices[:, 1] = num.repeat(source.get_patch_attribute('lon'), 4)

    for ip, p in enumerate(source.patches):
        vertices[ip*4:(ip+1)*4, 2:] = patch_outline(p, cs='xyz')[:-1, :]

    faces = num.zeros((n_patches, 5), dtype=num.int64)
    faces[:, :-1] = num.array([
        num.arange(ip, ip+4) for ip in range(0, n_patches * 4, 4)])
    faces[:, -1] = faces[:, 0]

    slip, times = source.get_delta_slip(
        delta=False,
        store=store)

    slip_norm = num.linalg.norm(slip, axis=2)
    slip_final = slip_norm[:, -1]

    outline = source.outline(cs='xyz')
    latlon = num.ones((5, 2)) * num.array([source.lat, source.lon])
    patchverts = num.hstack((latlon, outline))
    face_outlines = patchverts[:-1, :]  # last vertex double

    geom = geometry.Geometry()
    geom.set_vertices(vertices)
    geom.set_faces(faces)
    geom.set_outlines([face_outlines])

    geom.times = times

    geom.event = source.pyrocko_event(
        store=store,
        target=gf.seismosizer.Target(interpolation='nearest_neighbor'))

    geom.add_property(
        name=table.Header(
            name='final slip',
            label='final slip',
            sub_headers=[]),
        values=slip_final)

    geom.add_property(
        name=table.Header(
            name='slip (t)',
            label='slip (t)',
            sub_headers=['' for i in range(geom.times.shape[0])]),
        values=slip_norm)

    return geom


for i, source in enumerate(sources):
    geom = pdr_to_geometry(source)
    geom.dump(filename='../sparrow/source{}_geom.yaml'.format(
        chr(ord('@')+(i+1))))
