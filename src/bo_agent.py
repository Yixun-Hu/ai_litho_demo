"""
Bayesian Optimization Agent

This module implements a Bayesian optimization-based agent for intelligent process parameter search
in the closed-loop system.

Core Idea:
Bayesian optimization is an efficient black-box optimization method, particularly suitable for
objective functions with expensive evaluation costs.
It constructs a surrogate model (usually Gaussian Process) to approximate the true objective function,
and uses an acquisition function to decide the next most valuable evaluation point.

Key Components:
1. Gaussian Process (GP): 
   As the surrogate model, GP can not only predict the value of the objective function at any point,
   but also provide prediction uncertainty (variance).

2. Acquisition Function:
   This implementation uses Expected Improvement (EI) as the acquisition function.
   EI effectively balances Exploration and Exploitation.

References:
- BoTorch: https://botorch.org/
- Guler et al. (2021). Bayesian optimization for Tuning Lithography Processes.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class BayesianOptimizationAgent:
    """
    Bayesian Optimization Agent
    
    This agent is responsible for intelligently selecting the next set of candidate parameters
    to evaluate within a given parameter space.
    
    Attributes:
        param_bounds (dict): Search bounds for parameters
        X (torch.Tensor): Evaluated parameter points
        Y (torch.Tensor): Corresponding objective function values
        best_value (float): Best objective value found so far
        best_params (dict): Current best parameters
    """
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize Bayesian Optimization Agent
        
        Args:
            param_bounds: Parameter bounds dictionary, format: {'param_name': (lower, upper)}
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.dim = len(self.param_names)
        
        # Store observation data
        self.X = None  # Parameter points (N x D)
        self.Y = None  # Objective values (N x 1)
        
        # Best result tracking
        self.best_value = float('inf')
        self.best_params = None
        
        # History records
        self.history = {
            'params': [],
            'values': [],
            'best_values': []
        }
        
        # GP model parameters
        self.length_scale = 0.5
        self.noise_var = 1e-4
        
    def _normalize_params(self, params: Dict[str, float]) -> torch.Tensor:
        """
        Normalize parameters to [0, 1] interval
        
        Args:
            params: Original parameter dictionary
            
        Returns:
            Normalized parameter tensor
        """
        normalized = []
        for name in self.param_names:
            lower, upper = self.param_bounds[name]
            value = (params[name] - lower) / (upper - lower)
            normalized.append(value)
        return torch.tensor(normalized, dtype=torch.float64)
    
    def _denormalize_params(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Convert normalized parameters back to original scale
        
        Args:
            x: Normalized parameter tensor
            
        Returns:
            Parameter dictionary in original scale
        """
        params = {}
        for i, name in enumerate(self.param_names):
            lower, upper = self.param_bounds[name]
            params[name] = lower + x[i].item() * (upper - lower)
        return params
    
    def _rbf_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        RBF (Radial Basis Function) kernel function
        
        Args:
            X1: First set of points (N1 x D)
            X2: Second set of points (N2 x D)
            
        Returns:
            Kernel matrix (N1 x N2)
        """
        # Compute squared distances
        X1_sq = (X1 ** 2).sum(dim=1, keepdim=True)
        X2_sq = (X2 ** 2).sum(dim=1, keepdim=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        
        # RBF kernel
        K = torch.exp(-0.5 * dist_sq / (self.length_scale ** 2))
        return K
    
    def _gp_predict(self, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gaussian Process prediction
        
        Args:
            X_test: Test points (M x D)
            
        Returns:
            mean: Predicted mean (M,)
            var: Predicted variance (M,)
        """
        if self.X is None or len(self.X) == 0:
            # Return prior when no observation data
            return torch.zeros(len(X_test), dtype=torch.float64), torch.ones(len(X_test), dtype=torch.float64)
        
        # Compute kernel matrices
        K = self._rbf_kernel(self.X, self.X)
        K_s = self._rbf_kernel(self.X, X_test)
        K_ss = self._rbf_kernel(X_test, X_test)
        
        # Add noise
        K = K + self.noise_var * torch.eye(len(self.X), dtype=torch.float64)
        
        # Cholesky decomposition for solving
        try:
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(self.Y, L)
            v = torch.linalg.solve_triangular(L, K_s, upper=False)
        except:
            # Use more stable method if Cholesky decomposition fails
            K_inv = torch.linalg.pinv(K)
            alpha = K_inv @ self.Y
            v = K_s
        
        # Predicted mean and variance
        mean = (K_s.T @ alpha).squeeze()
        var = torch.diag(K_ss) - (v ** 2).sum(dim=0)
        var = torch.clamp(var, min=1e-6)  # Ensure variance is positive
        
        return mean, var
    
    def _expected_improvement(self, X_test: torch.Tensor, xi: float = 0.01) -> torch.Tensor:
        """
        Compute Expected Improvement (EI) acquisition function
        
        Args:
            X_test: Candidate points (M x D)
            xi: Exploration parameter
            
        Returns:
            EI values (M,)
        """
        mean, var = self._gp_predict(X_test)
        std = torch.sqrt(var)
        
        # Current best value
        if self.Y is not None and len(self.Y) > 0:
            f_best = self.Y.min()
        else:
            f_best = torch.tensor(0.0, dtype=torch.float64)
        
        # Compute EI
        improvement = f_best - mean - xi
        Z = improvement / std
        
        # Standard normal distribution CDF and PDF
        normal = torch.distributions.Normal(0, 1)
        Phi = normal.cdf(Z)
        phi = torch.exp(normal.log_prob(Z))
        
        EI = improvement * Phi + std * phi
        EI = torch.where(std > 1e-6, EI, torch.zeros_like(EI))
        
        return EI
    
    def suggest_next_params(self, n_candidates: int = 1000) -> Dict[str, float]:
        """
        Suggest next set of parameters to evaluate
        
        Uses random sampling + EI maximization strategy to select next evaluation point.
        
        Args:
            n_candidates: Number of random candidate points
            
        Returns:
            Suggested parameter dictionary
        """
        # Generate random candidate points
        candidates = torch.rand(n_candidates, self.dim, dtype=torch.float64)
        
        # Compute EI values
        ei_values = self._expected_improvement(candidates)
        
        # Select point with maximum EI
        best_idx = ei_values.argmax()
        best_candidate = candidates[best_idx]
        
        # Convert back to original scale
        suggested_params = self._denormalize_params(best_candidate)
        
        return suggested_params
    
    def update(self, params: Dict[str, float], value: float):
        """
        Update agent with new observation data
        
        Args:
            params: Evaluated parameters
            value: Corresponding objective function value
        """
        # Normalize parameters
        x = self._normalize_params(params).unsqueeze(0)
        y = torch.tensor([[value]], dtype=torch.float64)
        
        # Update data
        if self.X is None:
            self.X = x
            self.Y = y
        else:
            self.X = torch.cat([self.X, x], dim=0)
            self.Y = torch.cat([self.Y, y], dim=0)
        
        # Update best result
        if value < self.best_value:
            self.best_value = value
            self.best_params = params.copy()
        
        # Record history
        self.history['params'].append(params.copy())
        self.history['values'].append(value)
        self.history['best_values'].append(self.best_value)
    
    def get_initial_points(self, n_points: int = 5) -> List[Dict[str, float]]:
        """
        Generate initial sampling points (Latin Hypercube Sampling)
        
        Args:
            n_points: Number of initial points
            
        Returns:
            List of initial parameter points
        """
        points = []
        
        # Simplified Latin Hypercube Sampling
        for i in range(n_points):
            params = {}
            for name in self.param_names:
                lower, upper = self.param_bounds[name]
                # Divide interval into n_points segments, randomly sample within each segment
                segment_size = (upper - lower) / n_points
                segment_start = lower + i * segment_size
                params[name] = segment_start + np.random.random() * segment_size
            points.append(params)
        
        # Shuffle order
        np.random.shuffle(points)
        return points


def demo():
    """
    Demonstrate basic functionality of the Bayesian Optimization Agent
    """
    print("=" * 60)
    print("Bayesian Optimization Agent Demo")
    print("=" * 60)
    
    # Define parameter space
    param_bounds = {
        'dose': (15.0, 25.0),   # mJ/cm²
        'focus': (-0.2, 0.2)    # um
    }
    
    # Create agent
    agent = BayesianOptimizationAgent(param_bounds)
    
    # Define a simple objective function (simulating CD error)
    # Optimal point near dose=20, focus=0
    def objective(params):
        dose, focus = params['dose'], params['focus']
        # Simple quadratic function
        return (dose - 20.0)**2 + (focus * 50)**2
    
    print("\nObjective: Minimize f(dose, focus) = (dose-20)² + (focus×50)²")
    print(f"Parameter space: dose ∈ [{param_bounds['dose'][0]}, {param_bounds['dose'][1]}]")
    print(f"                 focus ∈ [{param_bounds['focus'][0]}, {param_bounds['focus'][1]}]")
    print(f"True optimum: dose=20.0, focus=0.0, f=0.0")
    
    # Initial sampling
    print("\n" + "-" * 60)
    print("Phase 1: Initial Sampling")
    print("-" * 60)
    
    initial_points = agent.get_initial_points(n_points=3)
    for i, params in enumerate(initial_points):
        value = objective(params)
        agent.update(params, value)
        print(f"  Initial point {i+1}: dose={params['dose']:.2f}, focus={params['focus']:.3f}, f={value:.4f}")
    
    # Bayesian optimization iterations
    print("\n" + "-" * 60)
    print("Phase 2: Bayesian Optimization Iterations")
    print("-" * 60)
    
    n_iterations = 10
    for i in range(n_iterations):
        # Get suggested next point
        suggested_params = agent.suggest_next_params()
        
        # Evaluate objective function
        value = objective(suggested_params)
        
        # Update agent
        agent.update(suggested_params, value)
        
        print(f"  Iteration {i+1}: dose={suggested_params['dose']:.2f}, focus={suggested_params['focus']:.3f}, "
              f"f={value:.4f}, best={agent.best_value:.4f}")
    
    # Output final results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Best parameters: dose={agent.best_params['dose']:.4f}, focus={agent.best_params['focus']:.4f}")
    print(f"Best objective value: {agent.best_value:.6f}")
    print(f"Total evaluations: {len(agent.history['values'])}")


if __name__ == "__main__":
    demo()
