"""
AI-Driven Closed-Loop Lithography Process Parameter Optimization System - Main Program

This program demonstrates a complete closed-loop optimization workflow:
1. Initialize lithography simulator and Bayesian optimization agent
2. Perform initial sampling to establish surrogate model
3. Iterative execution: Agent proposes parameters -> Simulation evaluation -> Model update
4. Output optimal parameters and optimization process visualization

This Demo shows how to apply AI methods to lithography process parameter optimization,
achieving the transition from "manual trial-and-error" to "intelligent search".
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litho_sim import DifferentiableLithoSim
from bo_agent import BayesianOptimizationAgent


class ClosedLoopOptimizer:
    """
    Closed-Loop Optimizer
    
    Integrates lithography simulator and Bayesian optimization agent to implement
    complete closed-loop optimization workflow.
    """
    
    def __init__(self, target_cd: float = 45.0):
        """
        Initialize Closed-Loop Optimizer
        
        Args:
            target_cd: Target critical dimension (nm)
        """
        self.target_cd = target_cd
        
        # Initialize lithography simulator
        self.simulator = DifferentiableLithoSim(target_cd=target_cd)
        
        # Define parameter search space
        self.param_bounds = {
            'dose': (15.0, 25.0),   # Exposure dose (mJ/cm²)
            'focus': (-0.15, 0.15)  # Defocus amount (um)
        }
        
        # Initialize Bayesian optimization agent
        self.agent = BayesianOptimizationAgent(self.param_bounds)
        
        # Optimization history
        self.optimization_history = []
        
    def evaluate(self, params: dict) -> dict:
        """
        Evaluate a set of parameters
        
        Args:
            params: Parameter dictionary containing 'dose' and 'focus'
            
        Returns:
            Evaluation result dictionary
        """
        result = self.simulator.forward(params['dose'], params['focus'])
        
        evaluation = {
            'params': params.copy(),
            'cd': result['cd'].item(),
            'cd_error': result['cd_error'].item(),
            'target_cd': self.target_cd
        }
        
        self.optimization_history.append(evaluation)
        return evaluation
    
    def run_optimization(self, n_initial: int = 5, n_iterations: int = 15):
        """
        Run complete closed-loop optimization workflow
        
        Args:
            n_initial: Number of initial sampling points
            n_iterations: Number of Bayesian optimization iterations
        """
        print("=" * 70)
        print("AI-Driven Closed-Loop Lithography Process Parameter Optimization System")
        print("=" * 70)
        print(f"\nObjective: Minimize CD error (Target CD = {self.target_cd} nm)")
        print(f"Parameter space:")
        print(f"  - Exposure Dose: {self.param_bounds['dose'][0]} - {self.param_bounds['dose'][1]} mJ/cm²")
        print(f"  - Defocus: {self.param_bounds['focus'][0]} - {self.param_bounds['focus'][1]} um")
        
        # Phase 1: Initial sampling
        print("\n" + "-" * 70)
        print("Phase 1: Initial Sampling (Building Surrogate Model)")
        print("-" * 70)
        
        initial_points = self.agent.get_initial_points(n_points=n_initial)
        
        print(f"{'#':<4} {'Dose (mJ/cm²)':<15} {'Focus (um)':<15} {'CD (nm)':<12} {'CD Error (nm)':<15}")
        print("-" * 70)
        
        for i, params in enumerate(initial_points):
            result = self.evaluate(params)
            self.agent.update(params, result['cd_error'])
            
            print(f"{i+1:<4} {params['dose']:<15.2f} {params['focus']:<15.4f} "
                  f"{result['cd']:<12.2f} {result['cd_error']:<15.4f}")
        
        # Phase 2: Bayesian optimization iterations
        print("\n" + "-" * 70)
        print("Phase 2: Bayesian Optimization Iterations (Intelligent Search)")
        print("-" * 70)
        print(f"{'#':<4} {'Dose (mJ/cm²)':<15} {'Focus (um)':<15} {'CD (nm)':<12} {'CD Error (nm)':<15} {'Best Error':<12}")
        print("-" * 70)
        
        for i in range(n_iterations):
            # Agent suggests next set of parameters
            suggested_params = self.agent.suggest_next_params()
            
            # Simulation evaluation
            result = self.evaluate(suggested_params)
            
            # Update agent
            self.agent.update(suggested_params, result['cd_error'])
            
            print(f"{n_initial+i+1:<4} {suggested_params['dose']:<15.2f} {suggested_params['focus']:<15.4f} "
                  f"{result['cd']:<12.2f} {result['cd_error']:<15.4f} {self.agent.best_value:<12.4f}")
        
        # Output final results
        print("\n" + "=" * 70)
        print("Optimization Results")
        print("=" * 70)
        print(f"Optimal parameters:")
        print(f"  - Dose = {self.agent.best_params['dose']:.4f} mJ/cm²")
        print(f"  - Focus = {self.agent.best_params['focus']:.6f} um")
        print(f"Minimum CD error: {self.agent.best_value:.4f} nm")
        print(f"Target CD achieved: {self.target_cd:.2f} ± {self.agent.best_value:.2f} nm")
        print(f"Total experiment count: {len(self.optimization_history)}")
        
        return self.agent.best_params, self.agent.best_value
    
    def save_results(self, output_dir: str = "results"):
        """
        Save optimization results and visualization plots
        
        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save optimization history data
        history_file = os.path.join(output_dir, "optimization_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)
        print(f"\nOptimization history saved to: {history_file}")
        
        # 2. Plot convergence curve
        self._plot_convergence(output_dir)
        
        # 3. Plot parameter search trajectory
        self._plot_search_trajectory(output_dir)
        
        # 4. Plot GP surrogate model
        self._plot_surrogate_model(output_dir)
    
    def _plot_convergence(self, output_dir: str):
        """Plot convergence curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(1, len(self.agent.history['best_values']) + 1)
        best_values = self.agent.history['best_values']
        all_values = self.agent.history['values']
        
        # Plot all evaluation points
        ax.scatter(iterations, all_values, c='blue', alpha=0.5, label='All Evaluations', s=50)
        
        # Plot best value curve
        ax.plot(iterations, best_values, 'r-', linewidth=2, label='Best Found')
        ax.scatter(iterations, best_values, c='red', s=80, zorder=5)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('CD Error (nm)', fontsize=12)
        ax.set_title('Optimization Convergence: CD Error vs Iteration', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Mark initial sampling and BO phases
        n_initial = len([h for h in self.optimization_history[:5]])  # Assume first 5 are initial sampling
        ax.axvline(x=n_initial + 0.5, color='green', linestyle='--', alpha=0.7)
        ax.text(n_initial/2, ax.get_ylim()[1]*0.9, 'Initial\nSampling', 
                ha='center', fontsize=10, color='green')
        ax.text((n_initial + len(iterations))/2, ax.get_ylim()[1]*0.9, 'Bayesian\nOptimization', 
                ha='center', fontsize=10, color='green')
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "convergence_plot.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Convergence curve saved to: {filepath}")
    
    def _plot_search_trajectory(self, output_dir: str):
        """Plot parameter search trajectory"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        doses = [h['params']['dose'] for h in self.optimization_history]
        focuses = [h['params']['focus'] for h in self.optimization_history]
        cd_errors = [h['cd_error'] for h in self.optimization_history]
        
        # Plot search trajectory
        scatter = ax.scatter(doses, focuses, c=range(len(doses)), 
                            cmap='viridis', s=100, alpha=0.8, edgecolors='black')
        
        # Plot connecting lines
        ax.plot(doses, focuses, 'k-', alpha=0.3, linewidth=1)
        
        # Mark start and end points
        ax.scatter(doses[0], focuses[0], c='green', s=200, marker='s', 
                  label='Start', edgecolors='black', linewidths=2, zorder=10)
        
        best_idx = cd_errors.index(min(cd_errors))
        ax.scatter(doses[best_idx], focuses[best_idx], c='red', s=200, marker='*', 
                  label='Best Found', edgecolors='black', linewidths=2, zorder=10)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Iteration', fontsize=12)
        
        ax.set_xlabel('Dose (mJ/cm²)', fontsize=12)
        ax.set_ylabel('Focus (um)', fontsize=12)
        ax.set_title('Parameter Search Trajectory', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_xlim(self.param_bounds['dose'])
        ax.set_ylim(self.param_bounds['focus'])
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "parameter_search.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Search trajectory plot saved to: {filepath}")
    
    def _plot_surrogate_model(self, output_dir: str):
        """Plot GP surrogate model prediction surface"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Create grid
        dose_range = np.linspace(self.param_bounds['dose'][0], self.param_bounds['dose'][1], 50)
        focus_range = np.linspace(self.param_bounds['focus'][0], self.param_bounds['focus'][1], 50)
        Dose, Focus = np.meshgrid(dose_range, focus_range)
        
        # Compute true objective function values (using simulator)
        Z_true = np.zeros_like(Dose)
        for i in range(Dose.shape[0]):
            for j in range(Dose.shape[1]):
                result = self.simulator.forward(Dose[i, j], Focus[i, j])
                Z_true[i, j] = result['cd_error'].item()
        
        # Plot true objective function
        ax1 = axes[0]
        contour1 = ax1.contourf(Dose, Focus, Z_true, levels=20, cmap='viridis')
        plt.colorbar(contour1, ax=ax1, label='CD Error (nm)')
        
        # Plot sampling points
        doses = [h['params']['dose'] for h in self.optimization_history]
        focuses = [h['params']['focus'] for h in self.optimization_history]
        ax1.scatter(doses, focuses, c='red', s=50, edgecolors='white', linewidths=1, zorder=5)
        
        ax1.set_xlabel('Dose (mJ/cm²)', fontsize=12)
        ax1.set_ylabel('Focus (um)', fontsize=12)
        ax1.set_title('True Objective Function (Lithography Simulation)', fontsize=12)
        
        # Plot GP surrogate model prediction
        ax2 = axes[1]
        
        # Use GP prediction
        import torch
        Z_pred = np.zeros_like(Dose)
        Z_std = np.zeros_like(Dose)
        
        for i in range(Dose.shape[0]):
            for j in range(Dose.shape[1]):
                params = {'dose': Dose[i, j], 'focus': Focus[i, j]}
                x = self.agent._normalize_params(params).unsqueeze(0)
                mean, var = self.agent._gp_predict(x)
                Z_pred[i, j] = mean.item()
                Z_std[i, j] = np.sqrt(var.item())
        
        contour2 = ax2.contourf(Dose, Focus, Z_pred, levels=20, cmap='viridis')
        plt.colorbar(contour2, ax=ax2, label='Predicted CD Error (nm)')
        
        # Plot sampling points
        ax2.scatter(doses, focuses, c='red', s=50, edgecolors='white', linewidths=1, zorder=5)
        
        ax2.set_xlabel('Dose (mJ/cm²)', fontsize=12)
        ax2.set_ylabel('Focus (um)', fontsize=12)
        ax2.set_title('GP Surrogate Model Prediction', fontsize=12)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, "surrogate_model.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Surrogate model plot saved to: {filepath}")


def main():
    """Main function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create closed-loop optimizer
    optimizer = ClosedLoopOptimizer(target_cd=45.0)
    
    # Run optimization
    best_params, best_value = optimizer.run_optimization(
        n_initial=5,      # Number of initial sampling points
        n_iterations=15   # Number of BO iterations
    )
    
    # Save results
    # Get project root directory from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")
    
    optimizer.save_results(results_dir)
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)
    print(f"\nPlease check output files in {results_dir}/ directory:")
    print("  - optimization_history.json: Complete optimization history data")
    print("  - convergence_plot.png: CD error convergence curve")
    print("  - parameter_search.png: Parameter space search trajectory")
    print("  - surrogate_model.png: GP surrogate model visualization")


if __name__ == "__main__":
    main()
