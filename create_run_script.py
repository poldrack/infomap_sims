# create shell script to run infomap_sims.py with different parameters

import os
import hashlib
import random
import numpy as np

# create directory for run scripts
os.makedirs("run_scripts", exist_ok=True)
nruns = 100

# create run script
# pilot testing suggested that the range from .15 to .4 captured
# the range from high to low accuracy in the clustering
with open("run_infomap_sims.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for noise_level in np.arange(0.15, 0.45, 0.05):
      for normalize in [True, False]:
        normalize_flag = '--normalize ' if normalize else ''
        for run in range(nruns):
            # create text hash of a random number
            random_hash = hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]
            f.write(
                f"python infomap_sims.py {normalize_flag}--noise_level {noise_level:.02f} --label {random_hash}\n"
            )
