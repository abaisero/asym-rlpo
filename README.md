# asym-porl
Asymmetric methods for partially observable reinforcement learning

## Installation

It is advised that you create an environment specifically for this code.
Assuming you have created and activated an environment, we next install
gym-gridverse (which is for now not yet available via PyPI).  We will install
gym-gridverse in edit mode to accomodate the ability to more easily pull
changes from the remote repository.

    cd /location/of/choice
    git clone git@github.com:abaisero/gym-gridverse.git
    cd gym-gridverse
    python -m pip install -e .

Next, move back to the location of this repository, and install the packages in
requirements.txt.

    cd /location/of/this/repo
    python -m pip install -r requirements.txt

As a quick check that everything was installed correctly, run main.py.

    python main.py
