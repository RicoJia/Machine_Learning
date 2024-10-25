#!/usr/bin/env python3

from ricomodels.utils.visualization import *

def test_training_timer():
    t = TrainingTimer()
    time.sleep(1)
    time_elapsed = t.lapse_time()
    assert abs(time_elapsed - 1) < 1e-1