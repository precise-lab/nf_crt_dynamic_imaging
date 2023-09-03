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

import requests
import shutil

def do_download(localfilename, my_url):
    print("Download ", my_url, " to ", localfilename)
    try:
        with requests.get(my_url, stream=True) as r:
            with open(localfilename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except:
        print("Failed to download ", localfilename, " from ", my_url )

if __name__ == "__main__":
    
    do_download('induced_pressure.mat', 'https://dataverse.harvard.edu/api/access/datafile/7350930')
    do_download('anatomy.mat', 'https://dataverse.harvard.edu/api/access/datafile/7350931')


