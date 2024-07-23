from config import *
import numpy as np

"""
This file is used to construct remaining test cases for LCR-rot-hop++ for each augmented data set.
"""
# Parse the content into a list of floats
remaining_pos_vector = np.load(f"remaining_test_indices_{FLAGS.year}")
outF= open(FLAGS.remaining_test_path, "w")

print(FLAGS.test_path)
with open(FLAGS.test_path, "r") as fd:
    for i, line in enumerate(fd):
        if i in remaining_pos_vector:
            outF.write(line)
outF.close()