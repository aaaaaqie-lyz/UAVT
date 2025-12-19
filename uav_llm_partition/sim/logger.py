"""Simple console logger."""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("uav_sim")

