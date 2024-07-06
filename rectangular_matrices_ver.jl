# Select your settings on lines 14-17, then press crtl + a followed by
# crtl + enter to run the entire script

using HomotopyContinuation
using LinearAlgebra
include("better_start_pair.jl");

## equivalent to norm(X, 2)^2
## squared frobenius norm, not spectral norm (for that, use opnorm instead)
f(X) = sum([X[i,j]^2 for i=1:size(X)[1], j=1:size(X)[2]]);

#### setting up equations ####

m, n = 3, 4;
### assumed r = 1 for this file
use_default_start_pair = true;
timeout = 30.0; # in seconds; MUST include a decimal point

seed = UInt32(rand(1:100));

@var A[1:m,1:n];
@var x[1:m], y[1:n];
critical_equations = differentiate(f(A - x*y'), vcat(x,y));
@var c[1:length(x)+1]; # affine chart to kill scale ambiguity
chart_equation = c' * vcat(x,1);
parametric_system = System(vcat(critical_equations[1:end-1], chart_equation);
                           parameters = vcat(vec(A), c));

# solve some problem instance for random A w/ monodromy

if use_default_start_pair
   monodromy_result =
      monodromy_solve(
         parametric_system;
         target_solutions_count=binomial(min(m,n),1), timeout=timeout);
else
   monodromy_result = 
      monodromy_solve(
         parametric_system, better_start_pair(parametric_system)...;
         target_solutions_count=binomial(min(m,n),1), timeout=timeout);
end
p0 = parameters(monodromy_result);
x0s = solutions(monodromy_result);

# now, solve system for "your favorite" parameters
rand_parameters = randn(length(p0));

homotopy_result =
   solve(parametric_system, x0s;
         start_parameters=p0, target_parameters=rand_parameters, seed=seed);
target_solutions = real_solutions(homotopy_result);
target_matrices = [sol[1:m] * sol[m.+(1:n)]' for sol in target_solutions];

# optimal solution using Eckart-Young
A1 = reshape(rand_parameters[1:m*n], m, n);
U,S,V = svd(A1);
opt = U[:,1] * S[1] * V[:,1]';

differences = [norm(opt - target_matrices[i]) for i = 1:length(target_matrices)]
obj_vals = [f(A1 - opt), [f(A1 - target_matrices[i]) for i = 1:length(target_matrices)]]

[evaluate(critical_equations, A => A1, x => sol[1:m],
                              y => sol[m.+(1:n)])
          for sol in target_solutions]

# Good next steps code: 
# 1. Working for mxn "input" matrices A ✓
# 2. Working for arbitrary-rank approximations (not just 1)
# 3. Generalize code to work for problem (1.1) in Li/Lim paper
   # One research question: is there a formula for all critical points of this problem (not just the optimum)?   