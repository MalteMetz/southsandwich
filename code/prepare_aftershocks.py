import os.path as op
import numpy as num

from pyrocko import model

path_as = ('/home/malde/Documents/Jobs_Praktika/PhD/publications/'
           'southsandwich/aftershocks')

events = model.load_events(
    filename=op.join(path_as, 'clustered_events.pf'))

cluster_tags = [t for ev in events for t in ev.tags if 'cluster' in t]
cluster_tags = [int(t.split(':')[1]) for t in cluster_tags]

clusters = num.unique(cluster_tags)

for cl in clusters:
    events_sorted = [
        events[i] for i in range(len(events)) if cluster_tags[i] == cl]

    model.dump_events(
        events_sorted,
        filename='../sparrow/aftershock_cluster_{}.yaml'.format(cl))
