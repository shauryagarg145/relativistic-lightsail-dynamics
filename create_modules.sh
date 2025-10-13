#!/bin/bash

# Load the appropriate modules so that pip can find your packages
module load python3/3.12.1

packages=(matplotlib numpy scipy pandas adaptive autograd grcwa nlopt torch torcwa multiprocess dill)
versionNums=(3.10.3 2.2.5 1.13.1 2.2.3 1.3.0 1.8.0 0.1.2 2.9.1 2.7.0 0.1.4.2 0.70.18 0.4.0)

for idx in ${!packages[*]}; do
        #pip3 install --prefix=$HOME/envs/${packages[$idx]}/${versionNums[$idx]} ${packages[$idx]}==${versionNums[$idx]} 
        if [[ "${packages[$idx]}" == "torch" ]]; then
                pip3 install --upgrade --force-reinstall --no-deps --user ${packages[$idx]}==${versionNums[$idx]} --index-url https://download.pytorch.org/whl/cpu
        else        
                pip3 install --upgrade --force-reinstall --no-deps --user ${packages[$idx]}==${versionNums[$idx]}
        fi
        # Remove module directory and replace with updated one every time this script is run to avoid mkdir error
        rm -r ~/privatemodules/${packages[$idx]}/; mkdir ~/privatemodules/${packages[$idx]}/
        echo -e '#%Module1.0\n' >| ~/privatemodules/${packages[$idx]}/${versionNums[$idx]}
        echo 'prepend-path    PYTHONPATH [getenv HOME]/envs/'"${packages[$idx]}"'/'"${versionNums[$idx]}"'/lib/python3.12/site-packages' >> ~/privatemodules/${packages[$idx]}/${versionNums[$idx]}
done