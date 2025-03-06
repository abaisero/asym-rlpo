import re

from asym_rlpo.envs import Environment
from asym_rlpo.policies import Policy
from asym_rlpo.policies.hardcoded.carflag import Carflag_HardcodedPolicy
from asym_rlpo.policies.hardcoded.cleaner import Cleaner_HardcodedPolicy
from asym_rlpo.policies.hardcoded.heavenhell import HeavenHell_HardcodedPolicy
from asym_rlpo.policies.hardcoded.shopping import Shopping_HardcodedPolicy


def make_hardcoded_policy(env_name: str, env: Environment) -> Policy:
    if match := re.match(r"POMDP-heavenhell_(\d+)-episodic-v0", env_name):
        size = int(match.group(1))
        return HeavenHell_HardcodedPolicy(size)

    if match := re.match(r"POMDP-shopping_(\d+)-episodic-v0", env_name):
        size = int(match.group(1))
        return Shopping_HardcodedPolicy(size)

    if env_name == "extra-cleaner-v0":
        return Cleaner_HardcodedPolicy()

    if env_name == "extra-car-flag-v0":
        return Carflag_HardcodedPolicy()

    raise NotImplementedError
