from gym.envs.registration import register

from .cleaner import EnvCleaner, EnvCleaner_Fix
from .dectiger import DecTiger, DecTiger_Fix
from .single_agent_wrapper import SingleAgentWrapper

register(
    id='extra-dectiger-v0',
    entry_point=lambda: SingleAgentWrapper(DecTiger_Fix(DecTiger())),
)

register(
    id='extra-cleaner-v0',
    entry_point=lambda: SingleAgentWrapper(EnvCleaner_Fix(EnvCleaner())),
)
