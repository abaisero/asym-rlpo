import re

from asym_rlpo.envs import Environment
from asym_rlpo.policies import Policy
from asym_rlpo.policies.hardcoded.heavenhell import HeavenHell_HardcodedPolicy


def make_hardcoded_policy(env_name: str, env: Environment) -> Policy:
    if match := re.match(r"POMDP-heavenhell_(\d+)-episodic-v0", env_name):
        size = int(match.group(1))
        return HeavenHell_HardcodedPolicy(size)

    raise NotImplementedError
