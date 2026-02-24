import numpy as np
from scipy.stats import norm
from abc import ABC
from ..curves.base import BaseYieldCurve

class TwoFactorGaussianModel(ABC): 
    """
    Two-Factor Gaussian (G2++) Short Rate Model.
    Dynamics: 
        dx_t = -a * x_t * dt + sigma * dW1_t
        dy_t = -b * y_t * dt + eta * dW2_t
        r_t = x_t + y_t + phi(t)
    Fits the initial term structure exactly.
    """
    def __init__(self, a, b, sigma, eta, rho, yield_curve: BaseYieldCurve):
        """
        Args:
            a, b (float): Mean reversion speeds for factors x and y.
            sigma, eta (float): Volatilities for factors x and y.
            rho (float): Correlation between the two Brownian motions.
            yield_curve (BaseYieldCurve): The initial market yield curve (e.g., NSS).
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.eta = eta
        self.rho = rho
        self.curve = yield_curve

    def V(self, t, T):
        """
        Calculates the variance V(t, T) of the integral of (x_u + y_u) (Equation 4.5).
        """
        dt = np.maximum(np.asanyarray(T) - t, 0.0)
        
        term1 = (self.sigma**2 / self.a**2) * (
            dt + (2 / self.a) * np.exp(-self.a * dt) 
            - (1 / (2 * self.a)) * np.exp(-2 * self.a * dt) - 3 / (2 * self.a)
        )
        
        term2 = (self.eta**2 / self.b**2) * (
            dt + (2 / self.b) * np.exp(-self.b * dt) 
            - (1 / (2 * self.b)) * np.exp(-2 * self.b * dt) - 3 / (2 * self.b)
        )
        
        term3 = (2 * self.rho * self.sigma * self.eta / (self.a * self.b)) * (
            dt + (np.exp(-self.a * dt) - 1) / self.a 
            + (np.exp(-self.b * dt) - 1) / self.b 
            - (np.exp(-(self.a + self.b) * dt) - 1) / (self.a + self.b)
        )
        
        return term1 + term2 + term3

    def B_mathcal(self, t, T, xt, yt):
        """
        Calculates the exponent term mathcal{B}(t, T) (Equation 4.4).
        """
        dt = np.maximum(np.asanyarray(T) - t, 0.0)
        
        part1 = 0.5 * (self.V(t, T) - self.V(0, T) + self.V(0, t))
        part2 = - ((1 - np.exp(-self.a * dt)) / self.a) * xt
        part3 = - ((1 - np.exp(-self.b * dt)) / self.b) * yt
        
        return part1 + part2 + part3

    def P(self, t, T, xt, yt):
        """
        Calculates the Zero-Coupon Bond value P(t, T) (Equation 4.3).
        """
        Pm_T = self.curve.P(0, T)
        
        Pm_t = self.curve.P(0, t) if t > 0 else 1.0
        
        return (Pm_T / Pm_t) * np.exp(self.B_mathcal(t, T, xt, yt))

    def zcb_option(self, t, T, xt, yt, T_expiry, K, option_type="call"):
        """
        Prices a European option on a Zero-Coupon Bond (Equations 4.6 & 4.7).
        Generalized for any evaluation time t.
        """
        tau_opt = np.maximum(T_expiry - t, 0.0)
        tau_bond = np.maximum(np.asanyarray(T) - T_expiry, 0.0)
        
        v1 = (self.sigma**2 / (2 * self.a**3)) * (1 - np.exp(-self.a * tau_bond))**2 * (1 - np.exp(-2 * self.a * tau_opt))
        v2 = (self.eta**2 / (2 * self.b**3)) * (1 - np.exp(-self.b * tau_bond))**2 * (1 - np.exp(-2 * self.b * tau_opt))
        v3 = (2 * self.rho * self.sigma * self.eta / (self.a * self.b * (self.a + self.b))) * (1 - np.exp(-self.a * tau_bond)) * (1 - np.exp(-self.b * tau_bond)) * (1 - np.exp(-(self.a + self.b) * tau_opt))
        
        sigma_p = np.sqrt(np.maximum(v1 + v2 + v3, 1e-12))
        
        Pt_T = self.P(t, T, xt, yt)
        Pt_Tcall = self.P(t, T_expiry, xt, yt)
        
        h = sigma_p / 2 + (1 / sigma_p) * np.log(Pt_T / (K * Pt_Tcall))
        
        if option_type == "call":
            return Pt_T * norm.cdf(h) - K * Pt_Tcall * norm.cdf(h - sigma_p)
        else:
            return K * Pt_Tcall * norm.cdf(sigma_p - h) - Pt_T * norm.cdf(-h)

    def rate_option(self, t, xt, yt, start_date, payment_dates, K_rate, nominal, option_type="cap"):
        """
        Prices a Cap or Floor product directly using the 2-factor ZCB options.
        """
        payment_dates = np.asanyarray(payment_dates)

        if start_date >= payment_dates[0]:
            raise ValueError("Start date must be strictly before the first payment date.")
        
        reset_dates = np.concatenate(([start_date], payment_dates[:-1]))
        future_mask = reset_dates >= t
        
        if not np.any(future_mask):
            return 0.0
            
        eff_resets = reset_dates[future_mask]
        eff_payments = payment_dates[future_mask] 
        
        deltas = eff_payments - eff_resets
        K_bonds = 1.0 / (1.0 + deltas * K_rate)
        
        zcb_option_type = "put" if option_type.lower() == "cap" else "call"
        
        option_prices = self.zcb_option(t, eff_payments, xt, yt, eff_resets, K_bonds, option_type=zcb_option_type)
        scale_factors = nominal * (1.0 + deltas * K_rate)
        
        return np.sum(option_prices * scale_factors)