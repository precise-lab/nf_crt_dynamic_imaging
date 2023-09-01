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

#Runs expirement with varying number of views per frame and fixed noise level

export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export OMP_NUM_THREADS=1

python -u image_reconstruction.py --out_folder VPT2/ --vpt 2 --ntB 2 --nrB 2
python create_animations.py --out_folder VPT2/ 

python -u image_reconstruction.py 
python create_animations.py 

python -u image_reconstruction.py  --out_folder VPT8/ --vpt 8 --ntB 3 --nrB 6
python create_animations.py --out_folder VPT8/ 

python -u image_reconstruction.py  --out_folder VPT16/ --vpt 16 --ntB 6 --nrB 6
python create_animations.py --out_folder VPT16/ 

python -u image_reconstruction.py --out_folder VPT32/ --vpt 32 --ntB 9 --nrB 10
python create_animations.py --out_folder VPT32/




