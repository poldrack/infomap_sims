# create shell script to run infomap_sims.py with different parameters

import os
import hashlib
import random
import numpy as np

# create directory for run scripts
os.makedirs("run_scripts", exist_ok=True)
nruns = 100

# create run script
with open("run_infomap_sims.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for noise_level in np.arange(0.1, 0.55, 0.05):
        for run in range(nruns):
            # create text hash of a random number
            random_hash = hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]
            f.write(
                f"python infomap_sims.py --noise_level {noise_level} --label {random_hash}\n"
            )
