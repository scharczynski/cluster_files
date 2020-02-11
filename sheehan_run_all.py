import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting


def run_script(cell_range):
    path_to_data = "/Users/stevecharczynski/workspace/data/sheehan/lin_pos_set/s11"
    data_processor = analysis.DataProcessor(
        path_to_data, cell_range)
    n_t = 2.
    solver_params = {
        "niter": 300,
        "stepsize": 500,
        "interval": 10,
        "method": "TNC",
        "use_jac": True,
        "T" : 1,
        "disp":False
    }
    bounds_pos = {
        "a_1": [10**-10, 1 / n_t],
        "ut": [0., 200.],
        "st": [10., 2000.],
        "a_0": [10**-10, 1 / n_t]
    }
    pipeline = analysis.Pipeline(cell_range, data_processor, [
                                "ConstVariable", "RelPosVariable","AbsPosVariable"])

    # pipeline.set_model_bounds("TimeVariableLength", bounds_t)
    pipeline.set_model_bounds("AbsPosVariable", bounds_pos)
    pipeline.set_model_bounds("RelPosVariable", bounds_pos)
    pipeline.set_model_bounds("ConstVariable",  {"a_0":[10**-10, 1]})
    pipeline.set_model_x0("AbsPosVariable", [1e-5, 20, 10, 1e-5])
    pipeline.set_model_x0("RelPosVariable", [1e-5, 20, 10, 1e-5])
    pipeline.set_model_x0("ConstVariable", [1e-5])
    # pipeline.show_rasters()
    import numpy as np
    import json
    with open(path_to_data+"/abs_pos.json", 'r') as f:
        abs_pos = np.array(json.load(f))
    with open(path_to_data+"/rel_pos.json", 'r') as f:
        rel_pos = np.array(json.load(f))
    with open(path_to_data+"/velocity.json", 'r') as f:
        velocity = np.array(json.load(f))
    pipeline.set_model_info("AbsPosVariable", "abs_pos", abs_pos, True)
    pipeline.set_model_info("RelPosVariable", "rel_pos", rel_pos, True)
    pipeline.set_model_info("AbsPosVariable", "velocity", velocity, True)
    pipeline.set_model_info("RelPosVariable", "velocity", velocity, True)
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.compare_models("ConstVariable", "RelPosVariable", 0.01, smoother_value=100)
    pipeline.compare_models("ConstVariable", "AbsPosVariable", 0.01, smoother_value=100)
    
if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)
