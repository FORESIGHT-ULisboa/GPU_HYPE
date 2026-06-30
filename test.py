import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpu_model.gpu_pso import GPU_PSO
from error.error_model import NewErrorModel
from conceptual.HYPE import HYPE
from gpu_model.functions import processBands
from gpu_model.functions2 import *

if __name__ == '__main__':

    obs = pd.read_csv("demo_model\Qobs.txt", sep="\t", header=0,
                    index_col="DATE", parse_dates=True)
    init_date = pd.to_datetime("2001-01-01")
    final_date = pd.to_datetime("2001-06-30")
    mask = (obs.index >= init_date) & (obs.index <= final_date)
    y = obs.loc[mask, ["3587"]]
    print("calibration records:", len(y), "|", y.index[0].date(), "\u2192", y.index[-1].date())

    calibration_parameters = ["wcfc", "rrcs1", "ttmp", "preccorr", "ttmp", "cmlt"]

    Model = HYPE(
        MultipleRuns=12,                 # parallel HYPE run folders
        calibration_parameters=calibration_parameters,
        random=True,                    # random initial population
        normalization=True,
        log=True,                       # default log-space sampling
        HYPEfolder=Path("demo_model\HYPE_setup_folder"),
        outfile="results/0003587.txt",  # HYPE basin-output for subbasin 50675
        parameter_bounds = { #Parameter bound
                    'wcfc': [0.0001, 1, True, 'substitute'], 
                    'rrcs1': [0.0001, 1, True, 'substitute'],
                    'rrcs2': [0.0001, 1, True, 'substitute'],
                    'preccorr': [0.0001, 1, True, 'substitute'],
                    'ttmp': [-3, 5, False, 'substitute'],
                    'cmlt': [0.001, 100, True, 'multiply'],
        },
    # Template: par_name : [lower_bound, upper_bound, log_space, 'substitute' or 'multiply']
    # Log is true if the parameters interval should be converted to log space, false if not.
    # Substitute - means all the values for that parameter are substituted, multiply - means they are multiplied by a unique factor.
    )

    Model.set_simulation_Dates(init_date, final_date)
    variables = len(Model.parList[0])
    print("number of optimisation variables:", variables)

    error_model = NewErrorModel(errorFunction="MSE")


    # (the figure folder is git-ignored). Without `save=`, fit raises at the end.
    os.makedirs("Simulations", exist_ok=True)
    save_fig = Path("Simulations/calibration.pplot")

    opt = GPU_PSO(
        modelObject=Model,
        errorObject=error_model,
        variables=variables,
        population=50,
        inertia=0.2, c1=0.3, c2=0.2, c3=0.001,
        pBins=5, partial=0.3,
        forcePositive=False, transformWeights=False,
        forceNonExceedance=0.01, displayEach=1,
        bands=[0.01, 0.025, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.975, 0.99]
    )

    result, performance = opt.fit(y, epochs=10, save=save_fig)
    print("\nresult keys:", list(result.keys()))
    print("calibrated population (members × variables):", result["parameters"].shape)
    print("fitness (members × [non_exceedance, log10 error]):", result["fitness"].shape)


    opt.modelObject.remove_tmpFiles()
    opt.save("examples/my_calibration.pkl")
    print("saved \u2192 examples/my_calibration.pkl")


    model = GPU_PSO.load("examples/my_calibration.pkl")

    obs = pd.read_csv("demo_model\Qobs.txt", sep="\t", header=0, index_col="DATE", parse_dates=True)

    index1 = pd.date_range("2001-07-01", "2001-12-31", freq="D")
    init_date = pd.to_datetime("2001-07-01")
    final_date = pd.to_datetime("2001-12-31")
    model.modelObject.init_tmpFiles()
    model.modelObject.set_simulation_Dates(init_date, final_date, b_init=pd.to_datetime("2001-01-01"))
    bands = model.opt['bands']
    predictions_ = model.predict() #goes to the decorator which only uses self
    predictions_ = processBands(predictions_, bands)
    predictions_df = pd.DataFrame(predictions_, index=index1, columns=bands)
    predictions_df.loc[:, 'Training'] = predictions_df.index < pd.to_datetime("2001-07-01")
    predictions_df.loc[:, 'Validation'] = predictions_df.index >= pd.to_datetime("2001-07-01")
    prec = pd.read_csv("demo_model\HYPE_setup_folder\Pobs.txt", sep="\t", header=0, index_col="DATE", parse_dates=True) 
    temp = pd.read_csv("demo_model\HYPE_setup_folder\Tobs.txt", sep="\t", header=0, index_col="DATE", parse_dates=True) 
    fig = plot_prediction(predictions_df, obs, temperature=temp, precipitation=prec)