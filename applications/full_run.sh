#Runs representatoin or embedding expirement
python representaiton.py
python create_animations.py --out_folder Representation_Results/
#Runs expirement with fixed number of views per frame and varying noise level
./variable_noise_level.sh
#Runs expirement with varying number of views per frame and fixed noise level
./variable_view.sh

