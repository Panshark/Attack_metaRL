import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_points(num):
    point_ani.set_data(x[num], y[num])
    return point_ani,