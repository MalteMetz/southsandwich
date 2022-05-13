import os.path as op
import numpy as num

from pyrocko import model, util

path_bp = ('/home/malde/Documents/Jobs_Praktika/PhD/publications/'
           'southsandwich/back_projection')

bp_fn = op.join(
    path_bp,
    '202108121832SOUTH_SANDWICH_ISLANDS_SEMBLANCE_RUPTURE_PATTERN.bp')

bp_lon, bp_lat, bp_t, bp_energy = num.loadtxt(bp_fn, skiprows=8, unpack=True)

events = []
for i in range(bp_lon.shape[0]):
    events.append(model.Event(
        lat=bp_lat[i],
        lon=bp_lon[i],
        depth=10e3,
        time=util.stt('2021-08-12 18:32:49') + bp_t[i],
        magnitude=bp_energy[i]))

model.dump_events(events, filename='../sparrow/back_projection.yaml')
