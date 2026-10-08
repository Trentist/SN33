"""Competitive SN33 miner package.

Enable with CUSTOM_MINER=1 (default when running neurons.miner after wiring).
Disable with CUSTOM_MINER=0 to fall back to stock MinerLib.
"""

from custom_miner.miner_lib import CompetitiveMinerLib

__all__ = ["CompetitiveMinerLib"]
