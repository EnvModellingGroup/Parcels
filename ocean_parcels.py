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
    ParticleSet,
    Field
)
from parcels import FiredrakeField


directory = "tests/test_data/output/hdf5/"
time = np.linspace(0, 50000, 51)
U = FiredrakeField.from_h5(directory, "Velocity2d", "uv_2d", "U", time=time)
V = FiredrakeField.from_h5(directory, "Velocity2d", "uv_2d", "V", time=time)

fieldset = FieldSet(U, V)

# generate our random points
all_points_x = [1, 100, 500]
all_points_y = [500, 450, 400]

def DeleteErrorParticle(particle, fieldset, time):
    if particle.state >= 40:  # deletes every particle that throws an error
        particle.delete()

pset = ParticleSet.from_list(fieldset=fieldset, pclass=ScipyParticle,
                             time=0,
                             lon=all_points_x,
                             lat=all_points_y)


output_file = pset.ParticleFile(name="Trajectory.zarr", outputdt=timedelta(seconds=10))
pset.execute([AdvectionRK4,DeleteErrorParticle],
             runtime=timedelta(seconds=6000),#, 16000, 171900, 1255885
             dt=timedelta(seconds=10),
             output_file=output_file)

ds = xr.open_zarr("Trajectory.zarr")

plt.plot(ds.lon.T, ds.lat.T, ".-")
plt.xlabel("Zonal distance [m]")
plt.ylabel("Meridional distance [m]")
plt.show()
