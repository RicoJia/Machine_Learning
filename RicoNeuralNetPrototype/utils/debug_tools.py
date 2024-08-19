#!/usr/bin/env python3

"""
In this file, we have a collection of utils for visualizations:
- Cost Visualization
- Gradient visualization
"""
import matplotlib.pyplot as plt
import numpy as np
import enum

class FCNDebuggerConfig(enum.Enum):
    compute_forward_gradients = True
    _log_batch_norm_stats = True
    
class FCNDebugger:
    def __init__(self, model, config: FCNDebuggerConfig):
        pass
    def record_and_calculate_backward_pass(self, loss):
        pass
    def _compute_gradient_norm(self):
        pass
    def _store_activations(self):
        # counting percentage of zero (dead neurons) in the output. Helpful especially for ReLu Activations
        pass
    def _log_batch_norm_stats(self):
        pass
    def plot_summary(self):
        # executes the registered functions
        pass

    def _log_softmax_inputs_outputs(self):
        pass

class CNNDebugger:
    def _get_feature_maps(self):
        # Grabs a feature map and stores it
        pass

    def _visualize_feature_maps(self):
        pass

if __name__ == "__main__":
    pass
    #sample usage:
    model = 1
    nn_config = FCNDebuggerConfig()
    nn_debugger = FCNDebugger(model=model, config=nn_config)
