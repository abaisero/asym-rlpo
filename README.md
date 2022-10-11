# asym-porl
Asymmetric methods for partially observable reinforcement learning

## Installation

It is advised that you create an environment specifically for this code.  This
repository depends on some of my other repositories that provide supporting
functionalities.  It is advised to install these in the following order:

* https://github.com/abaisero/rl-parsers
* https://github.com/abaisero/one-to-one
* https://github.com/abaisero/gym-pomdps
* https://github.com/abaisero/gym-gridverse

It is advised to install these in edit mode, to accomodate the ability to more
easily pull changes from the remote repositories.  For example, run the
following commands to install the `rl-parsers` code,

    cd /location/of/choice
    git clone git@github.com:abaisero/rl-parsers.git
    cd rl-parsers
    python -m pip install -e .

Repeat the above for all four prerequisite repositories.  Finally, move back to
the location of this repository, and install the packages in requirements.txt.

    cd /location/of/this/repo
    python -m pip install -r requirements.txt

As a quick check that everything was installed correctly, run main.py.

    python main.py
