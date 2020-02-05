import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json
import numpy as np


def run_script(cell_range):

    save_dir = "/projectnb/ecog-eeg/stevechar/sheehan_runs/random_lights_on_only/inbound/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/sheehan/random_lights_on_only/inbound/"
    data_processor = analysis.DataProcessor(
        path_to_data, cell_range)
    n_t = 2.
    solver_params = {
        "niter": 400,
        "stepsize": 500,
        "interval": 10,
        "method": "TNC",
        "use_jac": True,
        "T" : 1,
        "disp":False
    }
    bounds_dual = {
        "ut_a": [0., 90.],
        "st_a": [0.1, 90.],
        "a_0a": [10**-10, 1 / 3],
        "a_1a": [10**-10, 1 / 3],
        "ut_b": [0., 90.],
        "st_b": [0.1, 90.],
        "a_0b": [10**-10, 1 / 3],
        "a_1b": [10**-10, 1 / 3],
    }
    bounds_norm = {
        "a_1": [10**-10, 1 / n_t],
        "ut": [0., 90.],
        "st": [0.1, 90.],
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
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "ConstVariable", "RelPosVariable", "AbsPosVariable", "DualPeakedRel", "DualPeakedAbs"], save_dir=save_dir)
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "ConstVariable", "RelPosVariable", "AbsPosVariable"], save_dir=save_dir)
    # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    pipeline.set_model_bounds("AbsPosVariable", bounds_norm)
    pipeline.set_model_bounds("RelPosVariable", bounds_norm)
    pipeline.set_model_bounds("ConstVariable",  {"a_0":[10**-10, 1]})
    pipeline.set_model_bounds("DualPeakedRel", bounds_dual)
    pipeline.set_model_bounds("DualPeakedAbs", bounds_dual)
    pipeline.set_model_x0("DualPeakedRel", [20, 1, 1e-5, 1e-5, 20, 1, 1e-5, 1e-5])
    pipeline.set_model_x0("DualPeakedAbs", [20, 1, 1e-5, 1e-5, 20, 1, 1e-5, 1e-5])
    pipeline.set_model_x0("AbsPosVariable", [1e-5, 20, 1, 1e-5])
    pipeline.set_model_x0("RelPosVariable", [1e-5, 20, 1, 1e-5])
    pipeline.set_model_x0("ConstVariable", [1e-5])
    # pipeline.show_rasters()
    with open(path_to_data+"/abs_pos.json", 'r') as f:
        abs_pos = np.array(json.load(f))
    with open(path_to_data+"/rel_pos.json", 'r') as f:
        rel_pos = np.array(json.load(f))
    pipeline.set_model_info("AbsPosVariable", "abs_pos", abs_pos, True)
    pipeline.set_model_info("RelPosVariable", "rel_pos", rel_pos, True)
    pipeline.set_model_info("DualPeakedRel", "rel_pos", rel_pos, True)
    pipeline.set_model_info("DualPeakedAbs", "abs_pos", abs_pos, True)
    pipeline.fit_even_odd(solver_params=solver_params)
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.compare_even_odd("RelPosVariable", "DualPeakedRel", 0.01)
    pipeline.compare_even_odd("AbsPosVariable", "DualPeakedAbs", 0.01)
    pipeline.compare_even_odd("ConstVariable", "RelPosVariable", 0.01)
    pipeline.compare_even_odd("ConstVariable", "AbsPosVariable", 0.01)
    pipeline.compare_models("AbsPosVariable", "DualPeakedAbs", 0.01, smoother_value=100)
    pipeline.compare_models("RelPosVariable", "DualPeakedRel", 0.01, smoother_value=100)
    pipeline.compare_models("ConstVariable", "RelPosVariable", 0.01, smoother_value=100)
    pipeline.compare_models("ConstVariable", "AbsPosVariable", 0.01, smoother_value=100)

if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)
