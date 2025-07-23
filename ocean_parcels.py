import numpy as np
from datetime import timedelta
from parcels import (
    AdvectionRK4,
    FieldSet,
    ScipyParticle,
    ParticleSet,
    Field
)
from parcels import FiredrakeField
import pandas as pd
from firedrake.petsc import PETSc
import argparse
petsc_options = PETSc.Options()
petsc_options["options_left"] = False
import glob
import os

def DeleteErrorParticle(particle, fieldset, time):
    if particle.state >= 40:  # deletes every particle that throws an error
        particle.delete()

def main():

    parser = argparse.ArgumentParser(
         prog="oti_particles",
         description="""Run particles on a directory, from a start time, for 2 days"""
    )
    parser.add_argument(
        '-v', 
        '--verbose', 
        action='store_true', 
        help="Verbose output: mainly progress reports.",
        default=False
    )
    parser.add_argument(
        'model_dir',
        help='The model run directory, e.g. ../sims/base_case/'
    )
    parser.add_argument(
        'start_time',
        help="The start time (in model seconds) for the particle release",
        type=int,
    )

    args = parser.parse_args()
    verbose = args.verbose    
    model_dir = args.model_dir
    start_time = args.start_time

    if verbose:
        print("Sorting the firedrake outputs")
    directory = os.path.join(model_dir,"output/hdf5/")
    model_output_step = 900
    files = glob.glob(directory+"/Elevation*.h5")
    n_files = len(files)
    time = np.linspace(0, (n_files*model_output_step)-model_output_step, n_files)
    elev = FiredrakeField.from_h5(directory, "Elevation2d", "elev_2d", "Elev", time=time)
    U = FiredrakeField.from_h5(directory, "Velocity2d", "uv_2d", "U", time=time)
    V = FiredrakeField.from_h5(directory, "Velocity2d", "uv_2d", "V", time=time)
    U.add_wetting_and_drying(os.path.join(model_dir,"bathydg.h5"),"bathymetry", elev)
    V.add_wetting_and_drying(os.path.join(model_dir,"bathydg.h5"),"bathymetry", elev)
    fieldset = FieldSet(U, V)
    if verbose:
        print("\tI have "+str(n_files)+" outputs, over "+str(time[-1])+ "seconds")


    points = "/data/jh1889/oti_resolution/data/random_points.csv"
    start_points = pd.read_csv(points)
    all_points_x = start_points["X"]
    all_points_y = start_points["Y"]
    if verbose:
        print("Read in the starting points. I have "+str(len(all_points_x))+" points")

    pset = ParticleSet.from_list(fieldset=fieldset, pclass=ScipyParticle,
                                 time=start_time,
                                 lon=all_points_x,
                                 lat=all_points_y)


    output_file = pset.ParticleFile(
            name=os.path.join(model_dir,"Trajectory_"+str(start_time)+".zarr"), 
            outputdt=timedelta(seconds=240))
    pset.execute([AdvectionRK4,DeleteErrorParticle],
                 runtime=timedelta(seconds=2*24*60*60), # 2 days
                 dt=timedelta(seconds=120),
                 output_file=output_file)

if __name__ == "__main__":
    main()
