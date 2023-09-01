# Copyright (c) 2022, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the Neural Field CRT Dynamic Imaging Library. For more information and source code
# availability see https://github.com/precise-wustl/nf_crt_dynamic_imaging.
#
# Neural Field CRT Dynamic Imaging is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

#Runs expirement with fixed number of views per frame and varying noise level
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -u image_reconstruction.py --out_folder RNL0125/ --rnl 0.0125
python create_animations.py --out_folder RNL0125/ 

python -u image_reconstruction.py --out_folder RNL05/ --rnl 0.05
python create_animations.py --out_folder RNL05/ 

python -u image_reconstruction.py --out_folder RNL1/ --rnl 0.1
python create_animations.py --out_folder RNL1/ 

python -u image_reconstruction.py --out_folder RNL2/ --rnl 0.2
python create_animations.py --out_folder RNL2/ 





