from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZeroCouponAsset:
    maturity: float
    notional: float

    def value(self, curve_model, curve_factors, time_elapsed: float) -> float:
        """
        Value at time t of a zero-coupon asset.

        Remaining maturity = maturity - time_elapsed
        """
        ttm = self.maturity - time_elapsed
        if ttm <= 0.0:
            return float(self.notional)

        df = curve_model.discount_factor(curve_factors, ttm)
        return float(self.notional * df)
