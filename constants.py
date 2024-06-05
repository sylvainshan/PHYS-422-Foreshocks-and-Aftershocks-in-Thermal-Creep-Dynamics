import numpy as np


def load_tau_alpha(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return float(text.split(' ')[0])


# ======================================================================================================================

FAILURE_DATA_DIR = "failure_data/T=0.002"
FIGURE_DIR = "figures/"
SAVE_DIR = "data/"

# ======================================================================================================================

L = 512  # linear size of the system
D = 2  # dimension
TAU_ALPHA = load_tau_alpha("tau_alpha_L512_D2_0080to0089.txt")  # relaxation time

# ======================================================================================================================

VERBOSE = True
PLOT_CHECK = False
SAVE_PLOT = False

# ======================================================================================================================
# Test the value of tau_alpha
if __name__ == "__main__":
    print(f"TAU_ALPHA = {TAU_ALPHA}")
