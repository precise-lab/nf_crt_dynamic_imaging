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

#Downloads data set from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3DXS18
python download_dataset.py
#Runs representatoin or embedding expirement
python representaiton.py
python create_animations.py --out_folder Representation_Results/
#Runs expirement with fixed number of views per frame and varying noise level
./variable_noise_level.sh
#Runs expirement with varying number of views per frame and fixed noise level
./variable_view.sh

