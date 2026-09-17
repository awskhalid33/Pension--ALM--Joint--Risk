from __future__ import annotations

import os
import tempfile
from pathlib import Path


def get_pyplot(disable_plots: bool = False):
    """
    Return matplotlib.pyplot with a safe backend/config for local or headless runs.
    """
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path(tempfile.gettempdir()) / "pension_alm_mpl"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    import matplotlib

    if disable_plots and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg", force=True)
    elif "DISPLAY" not in os.environ and "MPLBACKEND" not in os.environ:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    return plt
