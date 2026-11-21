"""MouthKeys Linux. No telemetry, cloud transcription, or automatic updates."""

import os
from importlib.metadata import version

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(claw_is_pep526=False, is_pep484_tower=True))

# Capture at process startup so an update cannot relabel an older running service.
__version__ = version("mouthkeys-linux")

# Set before importing any model/download libraries, including from the CLI.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
