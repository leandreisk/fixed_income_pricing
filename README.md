# Fixed Income Securities Pricing

This repository provides a modular Python code for **Fixed Income pricing** and **Yield Curve modeling**. It implements an object-oriented architecture to handle various short-rate models (Vasicek, CIR, Hull-White) and static curve fitting methods (Nelson-Siegel-Svensson), designed for scalability and ease of extension to multi-factor models.

## Theoretical Background

### Yield Curve Modeling (Static)
The initial term structure of interest rates is modeled using the **Nelson-Siegel-Svensson (NSS)** parametric form:
$$y(t) = \beta_0 + \beta_1 \frac{1 - e^{-t/\tau_1}}{t/\tau_1} + \beta_2 \left( \frac{1 - e^{-t/\tau_1}}{t/\tau_1} - e^{-t/\tau_1} \right) + \beta_3 \left( \frac{1 - e^{-t/\tau_2}}{t/\tau_2} - e^{-t/\tau_2} \right)$$

### Stochastic Pricing Models
The library implements **One-Factor Affine Models** where the zero-coupon bond price $P(t,T)$ takes the form:
$$P(t, T) = A(t, T) e^{-B(t, T) r_t}$$

Supported dynamics include:
1.  **Vasicek**: Ornstein-Uhlenbeck process with constant parameters.
    $$dr_t = \kappa(\theta - r_t)dt + \sigma dW_t$$
2.  **Cox-Ingersoll-Ross (CIR)**: Square-root process ensuring positive rates.
    $$dr_t = \kappa(\theta - r_t)dt + \sigma \sqrt{r_t} dW_t$$
3.  **Hull-White**: Extension of Vasicek with time-dependent $\theta(t)$ to fit the initial term structure exactly.

## Features & Roadmap

- [x] **Core Architecture**: Abstract Base Classes for generic interest rate models.
- [x] **Curve Fitting**: Nelson-Siegel-Svensson (NSS) implementation.
- [x] **Affine Models**: Fully vectorized Vasicek and CIR implementations.
- [x] **Closed-Form Pricing**: Formulas for ZCBs and European options on ZCBs (Bond Puts/Calls).
- [x] **Coupon Bond Options**: Generic pricer using **Jamshidian's Decomposition** for any one-factor affine model.
- [x] **Hull-White Model**: Implementation with exact calibration to NSS input curves.
- [ ] **Calibration Engine**: Global optimizer to calibrate model parameters $(\kappa, \theta, \sigma)$ to market swaption volatilities.
- [ ] **Multi-Factor Support**: Architecture ready for G2++ (2-factor additive Gaussian).
- [ ] **Stochastic Volatility**: Future extension for Heston-Hull-White hybrid models.

## References

This implementation relies on standard results from financial mathematics literature:

* **Brigo, D., & Mercurio, F. (2006).** *Interest Rate Models - Theory and Practice.* Springer Finance.
* **Jamshidian, F. (1989).** *An Exact Bond Option Formula.* The Journal of Finance.