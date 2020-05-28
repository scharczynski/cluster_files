import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json

def run_script(cell_range):
    # path_to_data = "/Users/stevecharczynski/workspace/data/jay/ca1_time_cell_data/"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/jay/ca1_time_cells"
    # save_dir = "/Users/stevecharczynski/Desktop/test/"
    # # save_dir = "/projectnb/ecog-eeg/stevechar/ml_output/jay/ca1_time_cells/"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/all_sessions/{0}".format(session)

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range)
    # solver_params = {
    #     "niter": 1,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # n = 2
    # bounds_t = {
    #     "a_1": [0, 1 / n],
    #     "ut": [0, 9000],
    #     "st": [10, 6000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_dual = {
    #     "a_1": [0, 1/3],
    #     "a_2": [0, 1/3], 
    #     "ut_1": [0, 9000],
    #     "ut_2": [0, 9000],
    #     "st_1": [10, 6000],
    #     "st_2": [10, 6000],
    #     "a_0": [10**-10, 1/3]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1 / n]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "ConstVariable", "TimeVariableLength", "DualVariableLength"], save_dir=save_dir)
    # # pipeline = analysis.Pipeline(cell_range, data_processor, [
    # #     "ConstVariable", "RelPosVariable"], save_dir=save_dir)

    # # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("ConstVariable", bounds_c)
    # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("DualVariableLength", bounds_dual)
    # pipeline.set_model_x0("TimeVariableLength", [1e-5, 1000, 300, 1e-5])
    # pipeline.set_model_x0("ConstVariable", [1e-5])
    # pipeline.set_model_x0("DualVariableLength", [1e-5, 1e-5, 1000, 3000, 300, 300, 1e-5])
    # # pipeline.show_rasters()
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_even_odd("ConstVariable", "TimeVariableLength", 0.01)
    # pipeline.compare_even_odd("TimeVariableLength", "DualVariableLength", 0.01)
    # # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.compare_models("ConstVariable", "TimeVariableLength", 0.01, smoother_value=100)
    # # pipeline.compare_models("TimeVariableLength", "DualVariableLength", 0.01, smoother_value=100)


    # # path_to_data = "/Users/stevecharczynski/workspace/data/jay/lec/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/jay/lec/"
    # # save_dir = "/Users/stevecharczynski/workspace/data/jay/lec/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/jay/lec/"

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range)
    # solver_params = {
    #     "niter": 500,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False    
    # }
    # n = 2
    # bounds_t = {
    #     "a_1": [0, 1 / n],
    #     "ut": [0, 9000],
    #     "st": [10, 6000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_dual = {
    #     "a_1": [0, 1/3],
    #     "a_2": [0, 1/3], 
    #     "ut_1": [0, 9000],
    #     "ut_2": [0, 9000],
    #     "st_1": [10, 6000],
    #     "st_2": [10, 6000],
    #     "a_0": [10**-10, 1/3]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_smt = {
    #     "sigma": [0, 6000.],
    #     "mu": [0, 9000.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/2.],
    #     "a_0": [10**-10, 1/2.]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "ConstVariable", "TimeVariableLength", "DualVariableLength", "SigmaMuTauVariableLength"], save_dir=save_dir)
    # # pipeline = analysis.Pipeline(cell_range, data_processor, [
    # #     "ConstVariable", "RelPosVariable"], save_dir=save_dir)

    # # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("ConstVariable", bounds_c)
    # pipeline.set_model_bounds("SigmaMuTauVariableLength", bounds_smt)
    # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("DualVariableLength", bounds_dual)

    # # pipeline.set_model_bounds("LogNormalVariableLength", bounds_t)
    # pipeline.set_model_x0("SigmaMuTauVariableLength", [100, 1000, 300, 1e-5, 1e-5])
    # pipeline.set_model_x0("ConstVariable", [1e-5])
    # # pipeline.set_model_x0("LogNormalVariableLength", [1e-5, 1000, 100, 1e-5])
    # pipeline.set_model_x0("TimeVariableLength", [1e-5, 1000, 100, 1e-5])
    # pipeline.set_model_x0("DualVariableLength", [1e-5, 1e-5, 1000, 3000, 300, 300, 1e-5])

    # # pipeline.show_rasters()
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_even_odd("ConstVariable", "SigmaMuTauVariableLength", 0.01)
    # pipeline.compare_even_odd("ConstVariable", "TimeVariableLength", 0.01)
    # pipeline.compare_even_odd("TimeVariableLength", "SigmaMuTauVariableLength", 0.01)
    # pipeline.compare_even_odd("TimeVariableLength", "DualVariableLength", 0.01)

    # # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.compare_models("ConstVariable", "TimeVariableLength", 0.01, smoother_value=100)
    # pipeline.compare_models("TimeVariableLength", "SigmaMuTauVariableLength", 0.01, smoother_value=100)
    # pipeline.compare_models("ConstVariable", "SigmaMuTauVariableLength", 0.01, smoother_value=100)
    # pipeline.compare_models("TimeVariableLength", "DualVariableLength", 0.01, smoother_value=100)

    # pipeline.compare_models("TimeVariableLength", "DualVariableLength", 0.01, smoother_value=100)


    # path_to_data = "/Users/stevecharczynski/workspace/data/jay/lec/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/jay/lec_pyr"
    # save_dir = "/Users/stevecharczynski/workspace/data/jay/lec/"
    save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/jay/lec/"

    # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    data_processor = analysis.DataProcessor(
        path_to_data, cell_range)
    solver_params = {
        "niter": 500,
        "stepsize": 1000,
        "interval": 10,
        "method": "TNC",
        "use_jac": True,
        "T" : 1,
        "disp":False    
    }
    n = 2
    bounds_t = {
        "a_1": [0, 1 / n],
        "ut": [0, 9000],
        "st": [10, 6000],
        "a_0": [10**-10, 1 / n]
    }
    bounds_inhib = {
        "a_1": [10e-10, 1 / n],
        "ut": [0, 9000],
        "st": [10, 6000],
        "a_0": [10**-10, 1 / n]
    }
    bounds_dual = {
        "a_1": [0, 1/3],
        "a_2": [0, 1/3], 
        "ut_1": [0, 9000],
        "ut_2": [0, 9000],
        "st_1": [10, 6000],
        "st_2": [10, 6000],
        "a_0": [10**-10, 1/3]
    }
    bounds_c = {
        "a_0": [10**-10, 1 / n]
    }
    bounds_smt = {
        "sigma": [0, 6000.],
        "mu": [0, 9000.],
        "tau": [20, 20000.],
        "a_1": [10**-10, 1/2.],
        "a_0": [10**-10, 1/2.]
    }
    # bounds_smt_inhib = {
    #     "sigma": [0, 6000.],
    #     "mu": [0, 9000.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/2.],
    #     "a_0": [10**-10, 1/2.]
    # }
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "ConstVariable", "SigmaMuTauVariableLength","InhibitSigmaMuTauJayLEC", "TimeVariableLength", "TimeInhibitVariableLength"], save_dir=save_dir)
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "ConstVariable", "RelPosVariable"], save_dir=save_dir)

    # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    pipeline.set_model_bounds("ConstVariable", bounds_c)
    pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    pipeline.set_model_bounds("TimeInhibitVariableLength", bounds_inhib)
    pipeline.set_model_bounds("InhibitSigmaMuTauJayLEC", bounds_smt)
    pipeline.set_model_bounds("SigmaMuTauVariableLength", bounds_smt)


    # pipeline.set_model_bounds("LogNormalVariableLength", bounds_t)
    pipeline.set_model_x0("ConstVariable", [1e-5])
    pipeline.set_model_x0("SigmaMuTauVariableLength", [100, 1000, 300, 1e-5, 1e-5])
    pipeline.set_model_x0("InhibitSigmaMuTauJayLEC", [100, 1000, 300, 1e-5, 1e-4])
    # pipeline.set_model_x0("LogNormalVariableLength", [1e-5, 1000, 100, 1e-5])
    pipeline.set_model_x0("TimeVariableLength", [1e-5, 1000, 100, 1e-5])
    pipeline.set_model_x0("TimeInhibitVariableLength", [1e-5, 1000, 100, 1e-4])

    # pipeline.show_rasters()
    pipeline.fit_even_odd(solver_params=solver_params)
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.compare_even_odd("ConstVariable", "TimeInhibitVariableLength", 0.01)
    pipeline.compare_even_odd("ConstVariable", "SigmaMuTauVariableLength", 0.01)
    pipeline.compare_even_odd("ConstVariable", "InhibitSigmaMuTauJayLEC", 0.01)
    pipeline.compare_even_odd("ConstVariable", "TimeVariableLength", 0.01)


    # pipeline.compare_even_odd("Const", "Time", 0.01)
    pipeline.compare_models("ConstVariable", "TimeVariableLength", 0.01, smoother_value=100)
    pipeline.compare_models("ConstVariable", "TimeInhibitVariableLength", 0.01, smoother_value=100)
    pipeline.compare_models("ConstVariable", "SigmaMuTauVariableLength", 0.01, smoother_value=100)
    pipeline.compare_models("ConstVariable", "InhibitSigmaMuTauJayLEC", 0.01, smoother_value=100)


# run_script(range(41, 42))
if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)