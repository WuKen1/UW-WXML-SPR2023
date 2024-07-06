# Select your settings on lines 15-19, then press crtl + a followed by
# crtl + enter to run the entire script

using HomotopyContinuation
using LinearAlgebra
include("better_start_pair.jl"); ## make sure this file and file whose name is
##                                  in quotes are both in the working directory;
##                                  use cd() if needed

## equivalent to norm(X, 2)^2
## squared frobenius norm, not spectral norm (for that, use opnorm instead)
f(X) = sum([X[i,j]^2 for i = 1:size(X)[1], j = 1:size(X)[2]]);

#### setting up equations ####
use_default_start_pair = false;
m, n = 4, 5;
# maximum rank permitted of matrix approximating A
r = 2;  @assert r <= min(m,n)
timeout = 300.0; # in seconds; MUST include a decimal point

@var A[1:m,1:n];
@var X[1:m, 1:r], Y[1:n, 1:r]; # we'll take the outer product of ith columns of x and y
critical_equations = differentiate(f(A - sum(X[:,i]*Y[:,i]' for i = 1:r)), [vec(X); vec(Y)]);

@var C[1:r, 1:m], D[1:r, 1:r]

# e.g. [[2,1], [3,2]], or [[i,1] for i=1:m], or [[i,i] for i=1:min(m,r)]
indices_removed_from_X = []; # TO BE DETERMINED
indices_removed_from_Y = []; # TO BE DETERMINED

@assert all([1 ≤ i ≤ m && 1 ≤ j ≤ r for (i,j) ∈ indices_removed_from_X])
@assert all([1 ≤ i ≤ n && 1 ≤ j ≤ r for (i,j) ∈ indices_removed_from_Y])

indices_removed = [[(j-1)*m + i for (i,j) in indices_removed_from_X];
                   [m*r + (j-1)*n + i for (i,j) in indices_removed_from_Y]];
#@assert length(indices_removed) == r^2
indices_kept = setdiff(1:(m+n)*r, indices_removed);

parametric_system = System(vcat(critical_equations[indices_kept],
                                vec(C*X - D)
                               );
                           parameters = vcat(vec(A), vec(C), vec(D)));

# sanity-checking the start pair
(x0, p0) = better_start_pair(parametric_system)
residual_threshold = 1e-12
sv_threshold = 1e-16
@assert(sv_threshold < minimum(svdvals(jacobian(parametric_system,x0,p0))))
@assert(residual_threshold > norm(evaluate(parametric_system,x0,p0)))

if use_default_start_pair
    monodromy_result =
        monodromy_solve(
            parametric_system;
            target_solutions_count=binomial(min(m,n),r), timeout=timeout);
else
    monodromy_result = 
        monodromy_solve(
            parametric_system, better_start_pair(parametric_system)...;
            target_solutions_count=binomial(min(m,n),r), timeout=timeout);
end
p0 = parameters(monodromy_result);
x0s = solutions(monodromy_result.results);

# now, solve system for "your favorite" parameters
rand_parameters = randn(length(p0));
seed = UInt32(rand(1:100));
homotopy_result = solve(parametric_system, x0s;
                        start_parameters=p0, target_parameters=rand_parameters, seed=seed)
target_solutions = real_solutions(homotopy_result);

xs = [reshape(target_solutions[i][1:m*r], m, r) for i = 1:length(target_solutions)];
ys = [reshape(target_solutions[i][m*r .+ (1:r*n)], n, r) for i = 1:length(target_solutions)];
xy_pairs = [(xs[i], ys[i]) for i = 1:length(target_solutions)]

# optimal solution using Eckart-Young
A1 = reshape(rand_parameters[1:m*n], m, n);
U,S,V = svd(A1; full=true);
opt = sum(U[:,i] * S[i] * V[:,i]' for i = 1:r);

# see if any of the computed critical points correspond to the point
# suggested by the Eckart-Young theorem
differences = [opt - x*y' for (x,y) in xy_pairs]

# see how closely the computed critical points approximate the original matrix;
# shouldn't necessarily expect these to be small
obj_vals = [f(A1 - opt), [f(A1 - x*y') for (x,y) in xy_pairs]]

# sanity check that computed solutions actually satisfy the critical equations
evaluate(critical_equations, A => A1, X => U[:,1:r] * diagm(sqrt.(S[1:r])),
                                      Y => V[:,1:r] * diagm(sqrt.(S[1:r])))
[evaluate(critical_equations, A => A1, X => x, Y => y) for (x,y) in xy_pairs]