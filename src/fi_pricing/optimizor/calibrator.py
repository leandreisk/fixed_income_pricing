import numpy as np
from scipy.optimize import minimize, differential_evolution

class ModelCalibrator:
    """
    A universal calibrator for financial models.
    Minimizes a given objective function using SciPy optimizers.
    """
    def __init__(self, method='L-BFGS-B', tol=1e-8, max_iter=1000):
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.optimization_result = None

    def calibrate(self, objective_func, initial_guess, bounds=None):
        """
        Runs the optimization.
        
        Args:
            objective_func (callable): The function to minimize. It must take an array 
                                       of parameters as its first argument.
            initial_guess (list/array): Starting parameters (Theta 0).
            bounds (list of tuples): (min, max) for each parameter.
            
        Returns:
            np.ndarray: The optimal parameters.
        """
        print(f"Starting calibration using {self.method}...")
        
        options = {'maxiter': self.max_iter, 'disp': True}

        if self.method == "DE" : 
            if bounds: 
                self.optimization_result = differential_evolution(
                    objective_func,
                    bounds=bounds,
                    maxiter=self.max_iter,
                    tol=self.tol,
                    seed=42,
                    disp=True
                )
            else: 
                ValueError("Differential Evolution needs bounds")
        else: 
            self.optimization_result = minimize(
                fun=objective_func,
                x0=initial_guess,
                method=self.method,
                bounds=bounds,
                tol=self.tol,
                options=options
            )
        
        if self.optimization_result.success:
            print("Calibration successful!")
            print(f"Final Loss: {self.optimization_result.fun:.4e}")
        else:
            print("Calibration failed or did not converge perfectly.")
            print(f"Message: {self.optimization_result.message}")
            
        return self.optimization_result.x