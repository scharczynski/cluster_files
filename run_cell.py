import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json


def run_script(cell_range, session):

    # path_to_data = '/Users/stevecharczynski/workspace/data/jay/2nd_file'
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/jay"
    # # x = np.full(95, 0)
    # # y = np.full(95, 8000)

    # # trial_lengths = np.array(list(zip(x,y)))
    # # with open('/Users/stevecharczynski/workspace/data/jay/2nd_file/trial_lengths.json', 'w') as f:
    # #     json.dump(trial_lengths.tolist(), f)
    # data_processor = DataProcessor(
    #     path_to_data, cell_range)
    # n = 2
    # solver_params = {
    #     "niter": 2,
    #     "stepsize": 10000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds_smt = {
    #     "sigma": [10, 10000],
    #     "mu": [-1000, 10000],
    #     "tau": [0.0002, 0.10],
    #     "a_1": [10**-7, 0.5],
    #     "a_0": [10**-7, 0.5]
    # }
    # bounds_t = {
    #     "a_1": [0, 1 / n],
    #     "ut": [-1000,10000],
    #     "st": [100, 10000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1 / n]
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0)
    # # pipeline.set_model_bounds("Time", bounds_t)
    # # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.set_model_bounds(["Time", "Const"], [bounds_t, bounds_c])
    # pipeline.set_model_x0(["Const", "Time"], [[1e-5], [1e-5, 100, 100, 1e-5]])
    # # pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    # pipeline.fit_even_odd(solver_params)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("Const", "Time", 0.01)
    # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.compare_models("Time", "SigmaMuTau", 0.01)

    # path_to_data = '/Users/stevecharczynski/workspace/data/brincat_miller'
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/brincat_miller/"
    # time_info = [500, 1750]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n = 2
    # mean_delta = 0.10 * (time_info[1] - time_info[0])
    # mean_bounds = (
    #     (time_info[0] - mean_delta),
    #     (time_info[1] + mean_delta))
    # solver_params = {
    #     "niter": 100,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds_smt = {
    #     "sigma": [10, 1000],
    #     "mu": [100, 2300],
    #     "tau": [10, 5000],
    #     "a_1": [10**-7, 0.5],
    #     "a_0": [10**-7, 0.5]
    # }
    # bounds = {
    #     "a_1": [0, 1 / n],
    #     "ut": [mean_bounds[0], mean_bounds[1]],
    #     "st": [10, 1000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "SigmaMuTau", "Time"], 0)
    # # pipeline = AnalysisPipeline(cell_range, data_processor, [
    # #                         "Time","SigmaMuTau"], 0)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("Time", "SigmaMuTau", 0.01)

    # path_to_data = '/Users/stevecharczynski/workspace/data/brincat_miller'
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/brincat_miller/"
    # time_info = [500, 1750]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n = 2
    # mean_delta = 0.10 * (time_info[1] - time_info[0])
    # mean_bounds = (
    #     (time_info[0] - mean_delta),
    #     (time_info[1] + mean_delta))
    # solver_params = {
    #     "niter": 5,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds_t = {
    #     "a_1": [0, 1 / n],
    #     "ut": [mean_bounds[0], mean_bounds[1]],
    #     "st": [10, 1000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1 / n]
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0)
    # # pipeline = AnalysisPipeline(cell_range, data_processor, [
    # #                         "Time","SigmaMuTau"], 0)
    # pipeline.set_model_bounds("Time", bounds_t)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("Const", "Time", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/salz"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/salz"
    # time_info = [1000, 21000]
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=time_info)
    # solver_params = {
    #     "niter": 2,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds = {
    #     "a_1": [0, 1 / 2],
    #     "ut": [0, 42000,],
    #     "st": [10, 50000],
    #     "a_0": [10**-10, 1 / 2]
    # }
    # bounds_c = {"a_0": [10**-10, 0.999]}
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.set_model_x0("Time", [1e-5, 500, 500, 1e-5])
    # pipeline.set_model_x0("Const", [1e-5])
    # pipeline.fit_even_odd(solver_params)
    # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.fit_all_models(solver_params)
    # pipeline.compare_models("Const", "Time", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/cromer"
    # # path_to_data = '/projectnb/ecog-eeg/stevechar/data/cromer'
    # # path_to_data = "/usr3/bustaff/scharcz/workspace/cromer/"
    # time_info = [400, 2000]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n_c = 5
    # solver_params = {
    #     "niter": 1,
    #     "stepsize": 100,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": False,
    # }
    # bounds_cat = {
    #     "ut": [0, 2400],
    #     "st": [10, 5000],
    #     "a_0": [10**-10, 1 / n_c],
    #     "a_1": [10**-10, 1 / n_c],
    #     "a_2": [10**-10, 1 / n_c],
    #     "a_3": [10**-10, 1 / n_c],
    #     "a_4": [10**-10, 1 / n_c]
    # }
    # n_cs = 3
    # bounds_cs = {
    #     "ut": [0, 2400],
    #     "st": [10, 5000],
    #     "a_0": [10**-10, 1 / n_cs],
    #     "a_1": [10**-10, 1 / n_cs],
    #     "a_2": [10**-10, 1 / n_cs],
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "CatSetTime","CatTime"], 0.01)
    # pipeline.set_model_bounds("CatSetTime", bounds_cs)
    # pipeline.set_model_bounds("CatTime", bounds_cat)
    # pipeline.set_model_info("CatSetTime", "pairs", [(1,2), (3,4)])
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("CatSetTime", "CatTime", 0.01)
    # pipeline.show_condition_fit("CatTime")
    # pipeline.show_condition_fit("CatSetTime")

    # path_to_data = "/Users/stevecharczynski/workspace/data/kim"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/kim"
    # time_info = [0, 4784]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n = 2
    # swarm_params = {
    #     "phip": 0.5,
    #     "phig": 0.5,
    #     "omega": 0.5,
    #     "minstep": 1e-10,
    #     "minfunc": 1e-10,
    #     "maxiter": 1000
    # }
    # bounds = {
    #     "a_1": [0, 1 / n],
    #     "ut": [-200, 5200],
    #     "st": [10, 10000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {"a_0": [10**-10, 0.999]}
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(1)
    # pipeline.compare_models("Const", "Time", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/cromer"
    # # path_to_data =  "/projectnb/ecog-eeg/stevechar/data/cromer"
    # # with open(path_to_data+'/number_of_trials.json', 'r') as f:
    # #     num = json.load(f)
    # # x = np.full(max(num), 400)
    # # y = np.full(max(num), 2000)

    # # trial_lengths = np.array(list(zip(x,y)))
    # # with open(path_to_data+'/trial_lengths.json', 'w') as f:
    # #     json.dump(trial_lengths.tolist(), f)
    
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[400,2000])
    # n_t = 2.
    # solver_params = {
    #     "niter": 1,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp": False,
    # }
    # bounds = {
    #     "a_1": [10**-10, 1 / n_t],
    #     "ut": [0., 6000.],
    #     "st": [10., 5000.],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # bounds_c = {"a_0": [10**-10, 0.999]}
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #                             "Time", "Const"], 0)
    # # pipeline.show_rasters()

    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.set_model_x0("Time", [1e-5, 400, 100, 1e-5])
    # pipeline.set_model_x0("Const", [1e-5])
    # # pipeline.fit_even_odd(solver_params)
    # # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("Const", "Time", p_value=0.01, smoother_value=1)

    


    # path_to_data = '/Users/stevecharczynski/workspace/data/cromer'
    # # path_to_data = "/usr3/bustaff/scharcz/workspace/cromer/"
    # time_info = [400, 2000]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n_c = 3
    # swarm_params = {
    #     "phip": 0.5,
    #     "phig": 0.7,
    #     "omega": 0.7,
    #     "minstep": 1e-10,
    #     "minfunc": 1e-10,
    #     "maxiter": 1000
    # }
    # bounds_cat = {
    #     "ut": [0, 2400],
    #     "st": [10, 5000],
    #     "a_0": [10**-10, 1 / n_c],
    #     "a_1": [10**-10, 1 / n_c],
    #     "a_2": [10**-10, 1 / n_c],
    # }
    # n_t = 2
    # bounds_t = {
    #     "a_1": [0, 1 / n_t],
    #     "ut": [0, 2400],
    #     "st": [10, 5000],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # # bounds_cat = ((0,2400), (10, 5000), (10**-10, 1 / n), (0, 1 / n),(0, 1 / n), (0, 1 / n), (0, 1 / n))
    # # bounds= ((0, 1 / n), (0,2400), (10, 5000), (10**-10, 1 / n))
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "CatSetTime", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds_t)
    # pipeline.set_model_bounds("CatSetTime", bounds_cat)
    # pipeline.set_model_info("CatSetTime", "pairs", [(1,2), (3,4)])
    # pipeline.fit_all_models(1)
    # pipeline.compare_models("Time", "CatSetTime", 0.01)
    # pipeline.show_condition_fit("CatSetTime")

    # path_to_data = '/Users/stevecharczynski/workspace/rui_fake_cells/mixed_firing'
    # time_info = RegionInfo(0, 2000)
    # data_processor = DataProcessor(
    #     path_to_data, cell_range, time_info=time_info)
    # n = 2
    # bounds = {
    #     "a_1": [0, 1 / n],
    #     "ut": [-500, 2500],
    #     "st": [0.01, 5000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {"a_0": [10**-10, 0.99]}
    # swarm_params = {
    #     "phip": 0.5,
    #     "phig": 0.5,
    #     "omega": 0.5,
    #     "minstep": 1e-10,
    #     "minfunc": 1e-10,
    #     "maxiter": 1000
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time"], 0, swarm_params)
    # pipeline.set_model_bounds("Time", bounds)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.fit_all_models(3)
    # pipeline.compare_models("Const", "Time", 0.01)

    # util.collect_data(cell_range, "log_likelihoods")
    # util.collect_data(cell_range, "model_comparisons")
    # util.collect_data(cell_range, "cell_fits")

    # path_to_data = "/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1"
    # # path_to_data =  "/projectnb/ecog-eeg/stevechar/data/cromer"
    # time_info = [0, 3993]
    # data_processor = DataProcessor(
    #     path_to_data, cell_range)
    # n_t = 2.
    # solver_params = {
    #     "niter": 10,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # bounds_t = {
    #     "a_1": [10**-10, 1 / n_t],
    #     "ut": [0., 2400.],
    #     "st": [10., 5000.],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                            "Const", "TimeVariableLength"], 0)

    # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("Const",  {"a_0":[10**-10, 1]})

    # # with open("/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1/trial_length.json", 'rb') as f:
    # #     trial_length = json.load(f)
    # # pipeline.set_model_info("TimeVariableLength", "trial_length", trial_length)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("Const", "TimeVariableLength", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/crcns/hc-2/hc2/ec013.527"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/s25"
    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 1076686])
    # n_t = 2.
    # solver_params = {
    #     "niter": 20,
    #     "stepsize": 50,
    #     "interval": 4,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":True
    # }
    # bounds_pos = {
    #     "ut_x": [20, 200],
    #     "st_x": [1., 30.],
    #     "ut_y": [50., 250.],
    #     "st_y": [1., 30.],
    #     "a_0": [10**-10, 1 / n_t],
    #     "a_1": [10**-10, 1 / n_t]
    # }

    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #                            "PlaceField", "Const"])

    # # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("PlaceField", bounds_pos)
    # pipeline.set_model_bounds("Const", {"a_0":[10**-10, 0.5]})
    # pipeline.set_model_x0("PlaceField", [100,10,100,10,1e-5, 1e-5])
    # pipeline.set_model_x0("Const", [1e-5])
    # import numpy as np
    # import json
    # with open(path_to_data+"/pos_minus_removed.json", 'rb') as f:
    #     pos = np.array(json.load(f))
    # pipeline.set_model_info("PlaceField", "pos", pos, False)
    # model_dict = pipeline.fit_all_models(solver_params=solver_params)
    # plotting.plot_summed_2d(data_processor.spikes_binned[8], [300,300], pos, model_dict["PlaceField"][8].fit)
    # pipeline.compare_models("Const", "PlaceField", 0.01, smoother_value=1000)

    # path_to_data = "/Users/stevecharczynski/workspace/data/bulkin/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/bolkan/clusters_files/"
    # data_processor = analysis.DataProcessor(path_to_data, cell_range, [0, 60000])
    # solver_params = {
    #     "niter": 300,
    #     "stepsize": 5000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # bounds_vel = {
    #     "a_1": [10**-10, 1 / 2],
    #     "ut": [0., 70000.],
    #     "st": [100, 80000.],
    #     "a_0": [10**-10, 1 / 2]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #                            "Const", "Time"])
    # pipeline.set_model_bounds("Time", bounds_vel)
    # pipeline.set_model_bounds("Const", {"a_0":[10**-10, 1]})
    # pipeline.set_model_x0("Time", [1e-5, 20000, 1000, 1e-5])
    # pipeline.set_model_x0("Const", [1e-5])
    # # pipeline.show_rasters()
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("Const", "Time", 0.001, smoother_value=1000)
    # pipeline.compare_even_odd("Const", "Time", 0.001)

    # path_to_data = '/Users/stevecharczynski/workspace/data/brincat_miller'
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/brincat_miller/"
    # time_info = [500, 1750]
    # save_dir = "/Users/stevecharczynski/Desktop/test/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_output/brincat_miller/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/all_sessions/{0}".format(session)

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=time_info)
    # n_t = 2.
    # solver_params = {
    #     "niter": 5,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # mean_delta = 0.10 * (time_info[1] - time_info[0])
    # mean_bounds = (
    #     (time_info[0] - mean_delta),
    #     (time_info[1] + mean_delta))
    # n = 2
    # bounds_t = {
    #     "a_1": [0, 1 / n],
    #     "ut": [mean_bounds[0], mean_bounds[1]],
    #     "st": [10, 1000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1 / n]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const", "Time"], save_dir=save_dir)
    # # pipeline = analysis.Pipeline(cell_range, data_processor, [
    # #     "ConstVariable", "RelPosVariable"], save_dir=save_dir)

    # # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.set_model_bounds("Time", bounds_t)
    # pipeline.set_model_x0("Time", [1e-5, 700, 300, 1e-5])
    # pipeline.set_model_x0("Const", [1e-5])
    # # pipeline.show_rasters()
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "Time", 0.01)
    # # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.compare_models("Const", "Time", 0.01, smoother_value=100)

    # # path_to_data = "/Users/stevecharczynski/workspace/data/jay/ca1_time_cell_data/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/jay/ca1_time_cells"
    # # save_dir = "/Users/stevecharczynski/Desktop/test/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_output/jay/ca1_time_cells/"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/all_sessions/{0}".format(session)

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
    # pipeline.compare_models("TimeVariableLength", "DualVariableLength", 0.01, smoother_value=100)


    # # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # # save_dir = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recog_trials/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/"

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 1500])
    # solver_params = {
    #     "niter": 1000,
    #     "stepsize": 100,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # bounds_smtstim = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 1500.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/5.],
    #     "a_2": [10**-10, 1/5.],
    #     "a_3": [10**-10, 1/5.],
    #     "a_4": [10**-10, 1/5.],
    #     "a_0": [10**-10, 1/5.]
    # }
    # bounds_smt = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 1500.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/2.],
    #     "a_0": [10**-10, 1/2.]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const","SigmaMuTau", "SigmaMuTauStim"], save_dir=save_dir)
    # pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauStim", bounds_smtstim)
    # pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # # with open("/Users/stevecharczynski/workspace/data/warden/recog_trials/info.json") as f:
    # with open("/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/info.json") as f:
    #     stims = json.load(f)
    #     stims = {int(k):v for k,v in stims.items()}
    # pipeline.set_model_info("SigmaMuTauStim", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_x0("SigmaMuTauStim", [0.01, 1000, 100, 1e-1, 1e-1,1e-1, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTau", [0.01, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("Const", [1e-1])
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStim", 0.01)
    # pipeline.compare_models("Const", "SigmaMuTau", 0.01, smoother_value=100)
    # pipeline.compare_models("SigmaMuTau", "SigmaMuTauStim", 0.01, smoother_value=100)

    # # path_to_data = "/Users/stevecharczynski/workspace/data/rossi_pool/a2/"
    # # save_dir = "/Users/stevecharczynski/workspace/data/rossi_pool/a2/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/rossi_pool/a2/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/rossi_pool/a2/"

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 3000])
    # solver_params = {
    #     "niter": 1000,
    #     "stepsize": 100,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # bounds_smtstim = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 3000.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/5.],
    #     "a_2": [10**-10, 1/5.],
    #     "a_0": [10**-10, 1/5.]
    # }
    # bounds_smt = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 3000.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/2.],
    #     "a_0": [10**-10, 1/2.]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const","SigmaMuTau", "SigmaMuTauStimRP"], save_dir=save_dir)
    # pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauStimRP", bounds_smtstim)
    # pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # # with open("/Users/stevecharczynski/workspace/data/rossi_pool/a2/info.json") as f:
    # with open("/projectnb/ecog-eeg/stevechar/data/rossi_pool/a2/info.json") as f:
    #     stims = json.load(f)
    #     stims = {int(k):v for k,v in stims.items()}
    # pipeline.set_model_info("SigmaMuTauStimRP", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_x0("SigmaMuTauStimRP", [0.01, 1000, 100, 1e-1, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTau", [0.01, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("Const", [1e-1])
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStimRP", 0.01)
    # pipeline.compare_models("Const", "SigmaMuTau", 0.01, smoother_value=100)
    # pipeline.compare_models("SigmaMuTau", "SigmaMuTauStimRP", 0.01, smoother_value=100)


    # # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # # save_dir = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recall_trials"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/"

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 3000])
    # n_t = 2.
    # solver_params = {
    #     "niter": 100,
    #     "stepsize": 100,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # n = 2
    # bounds_smtstim = {
    #     "sigma1": [0, 1000.],
    #     "sigma2": [0, 1000.],
    #     # "sigma": [0.1, 0.15],
    #     "mu1": [0, 3000.],
    #     "mu2": [0, 3000.],
    #     "tau1": [20, 20000.],
    #     "tau2": [20, 20000.],
    #     "a_1": [10**-10, 1/5.],
    #     "a_2": [10**-10, 1/5.],
    #     "a_3": [10**-10, 1/5.],
    #     "a_4": [10**-10, 1/5.],
    #     "a_0": [10**-10, 1/5.]
    # }
    # bounds_smt = {
    #     "sigma1": [0, 1000.],
    #     "sigma2": [0, 1000.],
    #     # "sigma": [0.1, 0.15],
    #     "mu1": [0, 3000.],
    #     "mu2": [0, 3000.],
    #     "tau1": [20, 20000.],
    #     "tau2": [20, 20000.],
    #     "a_1": [10**-10, 1/5.],
    #     "a_0": [10**-10, 1/5.]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "SigmaMuTauDual", "SigmaMuTauDualStim"], save_dir=save_dir)
    # # pipeline = analysis.Pipeline(cell_range, data_processor, [
    # #     "ConstVariable", "RelPosVariable"], save_dir=save_dir)

    # # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("SigmaMuTauDual", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauDualStim", bounds_smtstim)
    # # with open("/Users/stevecharczynski/workspace/data/warden/recall_trials/info.json") as f:
    # with open("/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/info.json") as f:
    #     stims = json.load(f)
    #     stims = {int(k):v for k,v in stims.items()}
    # pipeline.set_model_info("SigmaMuTauDualStim", "stim_identity", stims, per_cell=True)

    # pipeline.set_model_x0("SigmaMuTauDualStim", [0.01,0.01, 1000,2000, 100, 100, 1e-1, 1e-1,1e-1, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTauDual", [0.01,0.01, 1000,2000, 100, 100, 1e-1, 1e-1])
    # pipeline.fit_all_models(solver_params=solver_params)
    # # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.compare_models("SigmaMuTauDual", "SigmaMuTauDualStim", 0.01, smoother_value=100)


    # # path_to_data = "/Users/stevecharczynski/workspace/data/sheehan/iti/sep_pos/1/"
    # # save_dir = "/Users/stevecharczynski/workspace/data/sheehan/iti/sep_pos/1/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/sheehan_runs/iti/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/iti/"
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 10500])
    # n_t = 2.
    # solver_params = {
    #     "niter": 300,
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
    #     "ut": [-1000, 11000],
    #     "st": [10, 6000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const", "Time"], save_dir=save_dir)
    # # pipeline = analysis.Pipeline(cell_range, data_processor, [
    # #     "ConstVariable", "RelPosVariable"], save_dir=save_dir)

    # # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.set_model_bounds("Time", bounds_t)
    # pipeline.set_model_x0("Time", [1e-5, 1000, 300, 1e-5])
    # pipeline.set_model_x0("Const", [1e-5])
    # # pipeline.show_rasters(show=False)
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "Time", 0.01)
    # # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.compare_models("Const", "Time", 0.01, smoother_value=100)


    # path_to_data = "/Users/stevecharczynski/workspace/data/rossi_pool/a1/"
    # save_dir = "/Users/stevecharczynski/workspace/data/rossi_pool/a1/results/"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/jay/ca1_time_cells"
    # # save_dir = "/Users/stevecharczynski/Desktop/test/"

    # # save_dir = "/projectnb/ecog-eeg/stevechar/ml_output/jay/ca1_time_cells/"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/all_sessions/{0}".format(session)

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 5000])
    # n_t = 2.
    # solver_params = {
    #     "niter": 10,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": False,
    #     "T" : 1,
    #     "disp":False
    # }
    # n = 2
    # bounds_smt = {
    #     "sigma": [1e-4, 1.],
    #     "mu1": [0, 5000.],
    #     "mu2": [0, 5000.],
    #     "tau": [10, 5000.],
    #     "a_1": [10**-2, 1/3.],
    #     "a_2": [10**-2, 1/3.],
    #     "a_0": [10**-4, 1/3.]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const", "SigmaMuTauDual"], save_dir=save_dir)
    # # pipeline = analysis.Pipeline(cell_range, data_processor, [
    # #     "ConstVariable", "RelPosVariable"], save_dir=save_dir)

    # # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("SigmaMuTauDual", bounds_smt)
    # pipeline.set_model_bounds("Const", bounds_c)
    # with open("/Users/stevecharczynski/workspace/data/rossi_pool/a1/stim_identity.json") as f:
    #     stims = json.load(f)
    # pipeline.set_model_info("SigmaMuTauDual", "stim_identity", stims)

    # pipeline.set_model_x0("SigmaMuTauDual", [0.01, 1000, 4000, 300, 1e-1, 1e-1, 1e-1])
    # pipeline.set_model_x0("Const", [1e-5])
    # # pipeline.show_rasters(save=True)
    # pipeline.fit_all_models(solver_params=solver_params)
    # # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.compare_models("Const", "SigmaMuTauDual", 0.01, smoother_value=100)


    # path_to_data = "/Users/stevecharczynski/workspace/data/rossi_pool/a1/"
    # save_dir = "/Users/stevecharczynski/workspace/data/rossi_pool/a1/results/"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/jay/ca1_time_cells"
    # # save_dir = "/Users/stevecharczynski/Desktop/test/"

    # # save_dir = "/projectnb/ecog-eeg/stevechar/ml_output/jay/ca1_time_cells/"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/all_sessions/{0}".format(session)

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 5000])
    # n_t = 2.
    # solver_params = {
    #     "niter": 5,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # n = 2
    # bounds_smtstim = {
    #     "sigma1": [0, 1000.],
    #     "sigma2": [0, 1000.],
    #     # "sigma": [0.1, 0.15],
    #     "mu1": [0, 5000.],
    #     "mu2": [0, 5000.],
    #     "tau1": [20, 20000.],
    #     "tau2": [20, 20000.],
    #     "a_1": [10**-5, 1/3.],
    #     "a_2": [10**-5, 1/3.],
    #     "a_0": [10**-5, 1/3.]
    # }
    # bounds_smt = {
    #     "sigma1": [0, 1000.],
    #     "sigma2": [0, 1000.],
    #     # "sigma": [0.1, 0.15],
    #     "mu1": [0, 5000.],
    #     "mu2": [0, 5000.],
    #     "tau1": [20, 20000.],
    #     "tau2": [20, 20000.],
    #     "a_1": [10**-5, 1/3.],
    #     "a_0": [10**-5, 1/3.]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "SigmaMuTauDual", "SigmaMuTauDualStim"], save_dir=save_dir)
    # # pipeline = analysis.Pipeline(cell_range, data_processor, [
    # #     "ConstVariable", "RelPosVariable"], save_dir=save_dir)

    # # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    # pipeline.set_model_bounds("SigmaMuTauDualStim", bounds_smtstim)
    # pipeline.set_model_bounds("SigmaMuTauDual", bounds_smt)
    # with open("/Users/stevecharczynski/workspace/data/rossi_pool/a1/stim_identity.json") as f:
    #     stims = json.load(f)
    # pipeline.set_model_info("SigmaMuTauDualStim", "stim_identity", stims)

    # pipeline.set_model_x0("SigmaMuTauDualStim", [0.01,0.01, 1000,4000, 100, 100, 1e-1, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTauDual", [0.01,0.01, 1000,4000, 100, 100, 1e-1, 1e-1])

    # # pipeline.set_model_x0("Const", [1e-5])
    # # pipeline.show_rasters(save=True)
    # pipeline.fit_all_models(solver_params=solver_params)
    # # pipeline.compare_even_odd("Const", "Time", 0.01)
    # pipeline.compare_models("SigmaMuTauDual", "SigmaMuTauDualStim", 0.01, smoother_value=100)

    # path_to_data = "/Users/stevecharczynski/workspace/maxlikespy/examples/input_data/"
    # save_dir = "/Users/stevecharczynski/Desktop/test/"
    # data_processor = analysis.DataProcessor(path_to_data, cell_range, [0, 10000])
    # solver_params = {
    #     "niter": 5,
    #     "stepsize": 200,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # bounds_vel = {
    #     "a_1": [10e-10, 1 / 2],
    #     "ut": [-1000, 12000.],
    #     "st": [100, 20000.],
    #     "a_0": [10e-10, 1 / 2]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #                            "Const", "Time"], save_dir=save_dir)
    # pipeline.set_model_bounds("Time", bounds_vel)
    # pipeline.set_model_bounds("Const", {"a_0":[10**-10, 1]})
    # pipeline.set_model_x0("Time", [1e-5, 2000, 200, 1e-5])
    # pipeline.set_model_x0("Const", [1e-5])
    # # pipeline.show_rasters()
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("Const", "Time", 0.001, smoother_value=100)
    # pipeline.compare_even_odd("Const", "Time", 0.001)



    # path_to_data = "/Users/stevecharczynski/workspace/data/sheehan/lin_track_ON271719/lights1_move1/inbound/"
    # save_dir = "/Users/stevecharczynski/workspace/data/sheehan/lin_track_ON271719/lights1_move1/inbound/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/sheehan_runs/random_lights_on_only/inbound/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/random_lights_on_only/inbound/"

    save_dir = "/projectnb/ecog-eeg/stevechar/sheehan_runs/random_lights_on_only/outbound/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/random_lights_on_only/outbound/"


    # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    data_processor = analysis.DataProcessor(
        path_to_data, cell_range)
    n_t = 2.
    solver_params = {
        "niter": 400,
        "stepsize": 5000,
        "interval": 10,
        "method": "TNC",
        "use_jac": True,
        "T" : 1,
        "disp":False
    }
    bounds_dual = {
        "ut_a": [0., 100.],
        "st_a": [0.1, 100.],
        "a_0a": [10**-10, 1 / n_t],
        "a_1a": [10**-10, 1 / n_t],
        "ut_b": [0., 100.],
        "st_b": [0.1, 100.],
        "a_0b": [10**-10, 1 / n_t],
        "a_1b": [10**-10, 1 / n_t],
    }
    bounds_norm = {
        "a_1": [10**-10, 1 / n_t],
        "ut": [0., 100.],
        "st": [0.1, 100.],
        "a_0": [10**-10, 1 / n_t]
    }
    bounds_t = {
        "a_1": [10**-10, 1 / n_t],
        "ut": [0., 5000.],
        "st": [10., 5000.],
        "a_0": [10**-10, 1 / n_t]
    }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    # #     "ConstVariable", "RelPosVariable","DualPeakedRel", "AbsPosVariable"])
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "ConstVariable", "RelPosVariable", "AbsPosVariable", "DualPeakedRel"], save_dir=save_dir)
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "ConstVariable", "RelPosVariable", "AbsPosVariable"], save_dir=save_dir)
    # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    pipeline.set_model_bounds("AbsPosVariable", bounds_norm)
    pipeline.set_model_bounds("RelPosVariable", bounds_norm)
    pipeline.set_model_bounds("ConstVariable",  {"a_0":[10**-10, 1]})
    # pipeline.set_model_bounds("DualPeakedRel", bounds_dual)
    # pipeline.set_model_x0("DualPeakedRel", [20, 1, 1e-5, 1e-5, 20, 1, 1e-5, 1e-5])
    pipeline.set_model_x0("AbsPosVariable", [1e-5, 20, 1, 1e-5])
    pipeline.set_model_x0("RelPosVariable", [1e-5, 20, 1, 1e-5])
    pipeline.set_model_x0("ConstVariable", [1e-5])
    # pipeline.show_rasters()
    import numpy as np
    import json
    with open(path_to_data+"/abs_pos.json", 'r') as f:
        abs_pos = np.array(json.load(f))
    with open(path_to_data+"/rel_pos.json", 'r') as f:
        rel_pos = np.array(json.load(f))
    pipeline.set_model_info("AbsPosVariable", "abs_pos", abs_pos, True)
    pipeline.set_model_info("RelPosVariable", "rel_pos", rel_pos, True)
    # pipeline.set_model_info("DualPeakedRel", "rel_pos", rel_pos, True)
    pipeline.fit_even_odd(solver_params=solver_params)
    pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_even_odd("ConstVariable", "DualPeakedRel", 0.01)
    pipeline.compare_even_odd("ConstVariable", "RelPosVariable", 0.01)
    pipeline.compare_even_odd("ConstVariable", "AbsPosVariable", 0.01)
    # pipeline.compare_models("ConstVariable", "DualPeakedRel", 0.01, smoother_value=100)
    pipeline.compare_models("ConstVariable", "RelPosVariable", 0.01, smoother_value=100)
    pipeline.compare_models("ConstVariable", "AbsPosVariable", 0.01, smoother_value=100)

    # path_to_data = "/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1"
    # # path_to_data =  "/projectnb/ecog-eeg/stevechar/data/cromer"
    # # with open("/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1/trial_length_old.json", 'rb') as f:
    # #     trial_length = json.load(f)
    # # time_info = np.array(list(zip(np.zeros(len(trial_length), dtype=int), trial_length)))
    # # with open("/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1/trial_length_new.json", 'w') as f:
    # #     json.dump(time_info.tolist(), f)
    # data_processor = DataProcessor(
    #     path_to_data, cell_range)
    # n_t = 2.
    # solver_params = {
    #     "niter": 1,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds_pos = {
    #     "a_1": [10**-10, 1 / n_t],
    #     "ut": [0., 1100.],
    #     "st": [10., 2000.],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # bounds_t = {
    #     "a_1": [10**-10, 1 / n_t],
    #     "ut": [0., 2400.],
    #     "st": [10., 5000.],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                            "ConstVariable", "AbsPosVariable"], 0)

    # pipeline.set_model_bounds("AbsPosVariable", bounds_pos)
    # pipeline.set_model_bounds("ConstVariable",  {"a_0":[10**-10, 1]})

    # with open("/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1/x_abs.json", 'rb') as f:
    #     abs_pos = np.array(json.load(f))
    # with open("/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1/x_rel.json", 'rb') as f:
    #     rel_pos = np.array(json.load(f))
    # pipeline.set_model_info("AbsPosVariable", "abs_pos", abs_pos, True)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("ConstVariable", "AbsPosVariable", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1"
    # # path_to_data =  "/projectnb/ecog-eeg/stevechar/data/cromer"
    # # with open("/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1/trial_length_old.json", 'rb') as f:
    # #     trial_length = json.load(f)
    # # time_info = np.array(list(zip(np.zeros(len(trial_length), dtype=int), trial_length)))
    # # with open("/Users/stevecharczynski/workspace/data/sheehan/lin_track_s1/trial_length_new.json", 'w') as f:
    # #     json.dump(time_info.tolist(), f)
    # data_processor = DataProcessor(
    #     path_to_data, cell_range)
    # n_t = 2.
    # solver_params = {
    #     "niter": 1,
    #     "stepsize": 1000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds_pos = {
    #     "a_1": [10**-10, 1 / n_t],
    #     "ut": [0., 1100.],
    #     "st": [10., 2000.],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # bounds_t = {
    #     "a_1": [10**-10, 1 / n_t],
    #     "ut": [0., 2400.],
    #     "st": [10., 5000.],
    #     "a_0": [10**-10, 1 / n_t]
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "ConstVariable", "Const"], 0)

    # pipeline.set_model_bounds("Const", {"a_0":[10**-10, 1]})
    # pipeline.set_model_bounds("ConstVariable",  {"a_0":[10**-10, 1]})

    # pipeline.fit_all_models(solver_params=solver_params)

    # path_to_data = '/Users/stevecharczynski/workspace/data/cromer/'
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/brincat_miller/"
    # # x = np.full(95, 0)
    # # y = np.full(95, 8000)

    # # trial_lengths = np.array(list(zip(x,y)))
    # # with open('/Users/stevecharczynski/workspace/data/jay/2nd_file/trial_lengths.json', 'w') as f:
    # #     json.dump(trial_lengths.tolist(), f)
    # data_processor = DataProcessor(
    #     path_to_data, cell_range)
    # n = 2
    # solver_params = {
    #     "niter": 2,
    #     "stepsize": 10000,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    # }
    # bounds_smt = {
    #     "sigma": [10, 1000],
    #     "mu": [0, 2200],
    #     "tau": [10, 5000],
    #     "a_1": [10**-7, 0.5],
    #     "a_0": [10**-7, 0.5]
    # }
    # bounds_t = {
    #     "a_1": [0, 1 / n],
    #     "ut": [0,2200],
    #     "st": [10, 1000],
    #     "a_0": [10**-10, 1 / n]
    # }
    # bounds_c = {
    #     "a_0": [10**-10, 1 / n]
    # }
    # pipeline = AnalysisPipeline(cell_range, data_processor, [
    #                             "Const", "Time", "SigmaMuTau"], 0)
    # pipeline.set_model_bounds("Time", bounds_t)
    # pipeline.set_model_bounds("Const", bounds_c)
    # pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.compare_models("Const", "Time", 0.01)
    # pipeline.compare_models("Time", "SigmaMuTau", 0.01)

# run_script(range(3,4), "s23")
if __name__ == "__main__":
    session = sys.argv[1]
    # session = "bolkan"
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range, session)
    # session = sys.argv[-3]
    # cell_range = sys.argv[-2:]
    # cell_range = list(map(int, cell_range))
    # cell_range = range(cell_range[0], cell_range[1]+1)
    # run_script(cell_range, session)