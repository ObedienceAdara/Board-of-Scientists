"""Agent prompt catalogue.

Kept as a dedicated module so role modules no longer depend on a repository
root ``prompts.py`` file. The current prompt definitions are re-exported from
the preserved implementation until prompt ownership is split per role.
"""
from board_of_scientists._legacy.prompts import *
