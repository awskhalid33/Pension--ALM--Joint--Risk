from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ZeroCouponBond:
    """
    Zero-coupon bond paying 1 at maturity T.
    """
    maturity: float  # years

    def price(self, curve) -> float:
        return curve.discount_factor(self.maturity)

    def duration(self) -> float:
        # Macaulay duration of a zero-coupon bond = maturity
        return float(self.maturity)


@dataclass(frozen=True)
class CouponBond:
    """
    Fixed-rate annual coupon bond (simplified).
    """
    maturity: int
    coupon_rate: float  # annual coupon rate

    def price(self, curve) -> float:
        pv = 0.0
        for t in range(1, self.maturity + 1):
            coupon = self.coupon_rate
            if t == self.maturity:
                coupon += 1.0
            pv += coupon * curve.discount_factor(t)
        return float(pv)

    def duration(self, curve) -> float:
        pv = self.price(curve)
        weighted = 0.0
        for t in range(1, self.maturity + 1):
            coupon = self.coupon_rate
            if t == self.maturity:
                coupon += 1.0
            weighted += t * coupon * curve.discount_factor(t)
        return float(weighted / pv)
