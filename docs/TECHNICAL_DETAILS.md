# Technical Details Document

This document elaborates on the technical implementation, design decisions, and underlying theoretical foundations of each core module in the Demo.

## 1. Differentiable Lithography Simulator

### 1.1 Design Motivation

Traditional lithography simulators (such as Mentor Calibre, Synopsys S-Litho) are highly accurate but "black-box" tools that cannot provide gradient information. This limits the choice of optimization algorithms to gradient-free methods only (such as genetic algorithms, simulated annealing).

Our simulator is implemented in PyTorch with the following characteristics:

| Feature | Traditional Simulator | This Demo Simulator |
|---------|----------------------|---------------------|
| Gradient Information | Not available | Auto-computable |
| Optimization Methods | Gradient-free only | Supports gradient methods |
| Execution Speed | Minutes | Milliseconds |
| Accuracy | Industrial-grade | Simplified model |

### 1.2 Physical Model

#### Optical Model

The imaging of a lithography system can be described by the Hopkins equation. In this simplified version, we use Gaussian blur to approximate the effect of the Point Spread Function (PSF):

```
I(x,y) = |M(x,y) ⊗ PSF(x,y)|² × Dose
```

Where:
- `M(x,y)` is the mask pattern
- `PSF(x,y)` is approximated as a Gaussian function, with standard deviation related to defocus
- `Dose` is the exposure dose

**Defocus Effect Modeling**:
```python
sigma = base_sigma + |focus| × 10.0
```

The larger the defocus, the wider the PSF, and the more blurred the image.

#### Photoresist Model

Using Sigmoid function to simulate the threshold effect of photoresist:

```
P(x,y) = sigmoid(steepness × (I(x,y) - threshold))
```

Where:
- `steepness` controls the steepness of the threshold
- `threshold` is the development threshold

### 1.3 CD Measurement

Critical Dimension (CD) is measured by counting the line width at the center row of the printed image:

```python
soft_count = sum(sigmoid(20 × (profile - 0.5)))
cd = soft_count × pixel_size
```

Soft threshold (Sigmoid) is used instead of hard threshold (Step function) to maintain differentiability.

## 2. Bayesian Optimization Agent

### 2.1 Theoretical Foundation

Bayesian Optimization is a Sequential Model-based Optimization method, particularly suitable for:
- Objective functions with expensive evaluation costs
- Black-box objective functions (no analytical form)
- Moderate parameter space dimensions (typically <20 dimensions)

Its core idea is:
1. Use a surrogate model (usually Gaussian Process) to approximate the objective function
2. Use an acquisition function to decide the next evaluation point
3. Update the surrogate model after evaluation
4. Repeat until convergence

### 2.2 Gaussian Process Surrogate Model

Gaussian Process (GP) is a non-parametric Bayesian method that assumes the objective function is a sample path of a Gaussian process.

**Kernel Function Selection**: We use RBF (Radial Basis Function) kernel:

```
k(x, x') = exp(-||x - x'||² / (2l²))
```

Where `l` is the length scale parameter, controlling the smoothness of the function.

**Prediction Formula**: Given observation data `(X, Y)`, the prediction for a new point `x*` is:

```
μ(x*) = K(x*, X) × K(X, X)⁻¹ × Y
σ²(x*) = K(x*, x*) - K(x*, X) × K(X, X)⁻¹ × K(X, x*)
```

### 2.3 Expected Improvement Acquisition Function

Expected Improvement (EI) is defined as:

```
EI(x) = E[max(f_best - f(x), 0)]
```

For GP models, EI has an analytical form:

```
EI(x) = (f_best - μ(x)) × Φ(Z) + σ(x) × φ(Z)
```

Where:
- `Z = (f_best - μ(x)) / σ(x)`
- `Φ` is the CDF of the standard normal distribution
- `φ` is the PDF of the standard normal distribution

The advantage of EI is automatic balance between exploration and exploitation:
- When `μ(x)` is small (good predicted value), EI is large → **Exploitation**
- When `σ(x)` is large (high uncertainty), EI is large → **Exploration**

## 3. Closed-Loop Optimization Process

### 3.1 Algorithm Pseudocode

```
Algorithm: Closed-Loop Lithography Process Optimization

Input: 
  - Parameter bounds: {dose: [15, 25], focus: [-0.15, 0.15]}
  - Target CD: 45 nm
  - Max iterations: 20

Output:
  - Optimal parameters (dose*, focus*)
  - Minimum CD error

1. Initialize:
   - Create lithography simulator
   - Create Bayesian optimization agent

2. Initial Sampling (Latin Hypercube):
   for i = 1 to n_initial:
       params_i = sample_from_bounds()
       cd_error_i = simulator.evaluate(params_i)
       agent.update(params_i, cd_error_i)

3. Bayesian Optimization Loop:
   for i = 1 to n_iterations:
       # Propose next point using EI
       params_next = agent.suggest_next_params()
       
       # Evaluate via simulation
       cd_error = simulator.evaluate(params_next)
       
       # Update surrogate model
       agent.update(params_next, cd_error)
       
       # Check convergence
       if cd_error < tolerance:
           break

4. Return agent.best_params, agent.best_value
```

### 3.2 Initial Sampling Strategy

We use simplified Latin Hypercube Sampling (LHS) to generate initial points. LHS ensures that each parameter dimension is uniformly covered, making it more efficient than pure random sampling.

### 3.3 Convergence Criteria

This Demo uses a fixed iteration count as the termination condition. In practical applications, consider:
- CD error below threshold
- No improvement for N consecutive iterations
- Budget (experiment count) exhausted

## 4. Relationship with Existing Work

The technical approach of this Demo is aligned with the following academic works:

| Work | Contribution | Reference in This Demo |
|------|--------------|------------------------|
| TorchLitho [1] | Open-source differentiable lithography framework | Design philosophy of differentiable simulator |
| LithoBench [2] | AI lithography benchmark | Datasets and evaluation metrics |
| BoTorch [3] | Bayesian optimization framework | GP and EI implementation |
| Guler et al. [4] | BO for lithography tuning | Problem modeling and experiment design |

## 5. Limitations and Future Work

### Current Limitations

1. **Simulation Accuracy**: Simplified model cannot capture all physical effects (such as polarization, aberrations)
2. **Parameter Dimensions**: Only considers 2 parameters, actual processes may have 10+
3. **Single-Objective Optimization**: Only optimizes CD error, does not consider yield, process window, etc.

### Future Extensions

1. **Integrate TorchLitho**: Use more accurate physical models
2. **Multi-Objective Optimization**: Use Pareto optimization to simultaneously optimize multiple objectives
3. **Transfer Learning**: Leverage historical data to accelerate optimization of new processes
4. **Sim-to-Real**: Introduce real data to calibrate simulation bias

## References

[1] Chen, G., et al. (2024). Open-Source Differentiable Lithography Imaging Framework.
[2] Zheng, S., et al. (2023). LithoBench: Benchmarking AI Computational Lithography.
[3] Balandat, M., et al. (2020). BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
[4] Guler, S., et al. (2021). Bayesian optimization for Tuning Lithography Processes.
