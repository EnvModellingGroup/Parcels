import math
from datetime import timedelta
from operator import attrgetter
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import random
import numpy as np
from parcels import (
    AdvectionRK4,
    FieldSet,
    ScipyParticle,
    JITParticle,
    ParticleSet,
    Field
)
from parcels import FiredrakeField
import pandas as pd

points = "/home/jh1889/work/projects/PalaeoTides/oti_hires/10k_random_points_on_oti.csv"
directory = "/data/jh1889/oti_hires/sims/coarse/output/hdf5/"
time = np.linspace(0, 2765700, 3073)
elev = FiredrakeField.from_h5(directory, "Elevation2d", "elev_2d", "Elev", time=time)
U = FiredrakeField.from_h5(directory, "Velocity2d", "uv_2d", "U", time=time)
V = FiredrakeField.from_h5(directory, "Velocity2d", "uv_2d", "V", time=time)
U.add_wetting_and_drying("/data/jh1889/oti_hires/sims/coarse/bathymetrydg.h5","bathymetry", elev)
V.add_wetting_and_drying("/data/jh1889/oti_hires/sims/coarse/bathymetrydg.h5","bathymetry", elev)

fieldset = FieldSet(U, V)

# generate our random points
#all_points_x = [406149.,407183,407250]
#all_points_y = [7401297,7400243,7401391]
start_points = pd.read_csv(points)
all_points_x = start_points["X"]
all_points_y = start_points["Y"]


def DeleteErrorParticle(particle, fieldset, time):
    if particle.state >= 40:  # deletes every particle that throws an error
        particle.delete()

pset = ParticleSet.from_list(fieldset=fieldset, pclass=ScipyParticle,
                             time=315000,
                             lon=all_points_x,
                             lat=all_points_y)


output_file = pset.ParticleFile(name="Trajectory.zarr", outputdt=timedelta(seconds=120))
pset.execute([AdvectionRK4,DeleteErrorParticle],
             runtime=timedelta(seconds=172800),# 2 days
             dt=timedelta(seconds=120),
             output_file=output_file)

#ds = xr.open_zarr("Trajectory.zarr")
#
#plt.plot(ds.lon.T, ds.lat.T, ".-")
#plt.xlabel("Zonal distance [m]")
#plt.ylabel("Meridional distance [m]")
#plt.show()
