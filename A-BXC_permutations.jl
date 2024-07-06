# Select your settings on lines 16-20, then press crtl + a followed by
# crtl + enter to run the entire script
# (may need to substitute command key for ctrl on Mac)

using HomotopyContinuation
using LinearAlgebra
include("better_start_pair.jl"); ## make sure this file and file whose name is
##                                  in quotes are both in the working directory;
##                                  use cd() if needed
include("Li & Lim, Section 2.-2.1.jl");

## equivalent to norm(X, 2)^2
## squared frobenius norm, not spectral norm (for that, use opnorm instead)
f(X) = sum([X[i,j]^2 for i = 1:size(X)[1], j = 1:size(X)[2]]);

#### setting up equations ####
use_default_start_pair = true;
m, p, q, n = 2, 2, 2, 2; @assert m >= p && n >= q # (3,3,4,4,  2)
r = 1;  @assert 0 <= r <= min(p,q) # maximum rank permitted of X
timeout = 600.0; # in seconds; MUST include a decimal point

binomial(min(2,2),1)

@var A[1:m,1:n];
@var X[1:p, 1:r], Y[1:q, 1:r]; # we'll take the outer product of ith columns of x and y
@var B[1:m, 1:p], C[1:q, 1:n];
critical_equations = differentiate(f(A - B*X*Y'*C), [vec(X); vec(Y)]);

@var D[1:r, 1:p], E[1:r, 1:r]

# e.g. [[2,1], [3,2]], or [[i,1] for i=1:m], or [[i,i] for i=1:min(m,r)]
indices_removed_from_X = [[i,j] for i=1:r, j=1:r];
indices_removed_from_Y = [];

@assert all([1 ≤ i ≤ p && 1 ≤ j ≤ r for (i,j) ∈ indices_removed_from_X])
@assert all([1 ≤ i ≤ q && 1 ≤ j ≤ r for (i,j) ∈ indices_removed_from_Y])

indices_removed = [vec([(j-1)*p + i for (i,j) in indices_removed_from_X]);
                   vec([p*r + (j-1)*q + i for (i,j) in indices_removed_from_Y])];
indices_kept = setdiff(1:(p+q)*r, indices_removed);

parametric_system = System(vcat(critical_equations[indices_kept], vec(D*X - E));
                           parameters = vcat(vec.([A, B, C, D, E])...));
num_equations = length(expressions(parametric_system));
num_variables = length(expressions(parametric_system));
@assert(num_equations == num_variables)

# sanity-checking the start pair
(x0, p0) = find_start_pair(parametric_system);
residual_threshold = 1e-12;
sv_threshold = 1e-16;
@assert(sv_threshold < minimum(svdvals(jacobian(parametric_system,x0,p0))))
@assert(residual_threshold > norm(evaluate(parametric_system,x0,p0)))

## OFFLINE PHASE OF HC: Solve the problem for random parameters (over complex numbers)
if use_default_start_pair
    monodromy_result =
        monodromy_solve(
            parametric_system;
            target_solutions_count=binomial(min(p,q),r), timeout=timeout);
else
    monodromy_result = 
        monodromy_solve(
            parametric_system, better_start_pair(parametric_system)...;
            target_solutions_count=binomial(min(p,q),r), timeout=timeout);
end
p0 = parameters(monodromy_result);
x0s = solutions(monodromy_result.results);

# ONLINE PHASE: solve system for "your favorite" parameters
rand_parameters = randn(length(p0));
seed = UInt32(rand(1:100));
homotopy_result = solve(parametric_system, x0s; start_parameters=p0, target_parameters=rand_parameters, seed=seed)
target_solutions = real_solutions(homotopy_result);

A_0 = reshape(rand_parameters[1:m*n], m, n)
B_0 = reshape(rand_parameters[m*n .+ (1:m*p)], m, p)
C_0 = reshape(rand_parameters[(m*n + m*p) .+ (1:q*n)], q, n)

xs = [reshape(target_solutions[i][1:p*r], p, r) for i = 1:length(target_solutions)];
ys = [reshape(target_solutions[i][p*r .+ (1:r*q)], q, r) for i = 1:length(target_solutions)];
xy_pairs = [(xs[i], ys[i]) for i = 1:length(target_solutions)]

closed_form_sol = closed_form_solution(A_0, B_0, C_0, r);

# see if any of the computed critical points correspond to the point
# suggested by the closed-form solution from Li and Lim's paper
diff_in_norm = [norm(closed_form_sol - x*y') for (x,y) in xy_pairs]
relative_diff_in_norm = [(norm(closed_form_sol) - norm(x*y'))/norm(closed_form_sol) for (x,y) in xy_pairs]

index_best = findmin(relative_diff_in_norm)[2];
best_sol_pair = xy_pairs[index_best];

# see how closely the computed critical points approximate the original matrix;
# shouldn't necessarily expect these to be that small (i.e. 10^-16 will often be unrealistic)
obj_vals = [f(A_0 - B_0 * closed_form_sol * C_0),
            [f(A_0 - B_0 * x*y' * C_0) for (x,y) in xy_pairs]]

#rounded_matrices = [(num -> round(num, digits=4)).(x*y') for (x,y) in xy_pairs];
#for matrix in rounded_matrices
#    display(matrix)
#end

using Combinatorics
# try and see if any of the critical points can be expressed as some
# kind of permutations of the matrices that the closed-form solution is
# built from
function permute_columns(M, p)
    @assert length(p) == size(M)[2] "permuted index vector's length does not match # of columns/rows"
    @assert all(i ∈ p for i ∈ 1:size(M)[2]) "not all columns/rows are accounted for"
    return M[:,p]
end
function column_permutations(M)
    P = Combinatorics.permutations(1:size(M)[2]);
    return (p -> permute_columns(M, p)).(P)
end
permute_rows(M, p) = permute_columns(M', p)'
row_permutations(M) = transpose.(column_permutations(M'))

# For naming convention, refer to Li & Lim paper
V_B, sf, U_C, sz = solution_factors(A_0, B_0, C_0, r);

V_B_permutes = column_permutations(V_B) ∪ row_permutations(V_B);
S_B_inv_permutes = Diagonal.(Combinatorics.permutations(diag(sf[1])));
U_r_permutes = column_permutations(sf[2]) ∪ row_permutations(sf[2]);
Σ_r_permutes = Diagonal.(Combinatorics.permutations(diag(sf[3])));
V_r_permutes = column_permutations(sf[4]') ∪ row_permutations(sf[4]');
S_C_inv_permutes = Diagonal.(Combinatorics.permutations(diag(sf[5])));
U_C_permutes = column_permutations(U_C) ∪ row_permutations(U_C);

# There's gotta be a better way to write this...
candidates = Array{Tuple{Matrix,Matrix,Matrix,Matrix,Matrix,Matrix,Matrix}}(undef, 0);

are_all_critical_points = true; # until demonstrated otherwise
for (V_B, S_B_inv, U_r, Σ_r, V_r, S_C_inv, U_C) ∈ Iterators.product(V_B_permutes,
                                                            S_B_inv_permutes,
                                                            U_r_permutes,
                                                            Σ_r_permutes,
                                                            V_r_permutes,
                                                            S_C_inv_permutes,
                                                            U_C_permutes)
    #println()
    X_11 = S_B_inv * U_r * Σ_r * V_r' * S_C_inv;
    X_overall = V_B * [X_11               zeros(sz[1,2]...);
                       zeros(sz[2,1]...)  zeros(sz[2,2]...)] * U_C';
    is_a_critical_point = false;
    for (x,y) in xy_pairs[(1:end) .!= index_best]
        #display(x*y' - X_overall)
        if norm(x*y' - X_overall) < 10^-3
            is_a_critical_point = true;
            display(x*y' - X_overall)
            push!(candidates, (V_B, S_B_inv, U_r, Σ_r, V_r, S_C_inv, U_C))
            #break;
        end
    end
    if !is_a_critical_point
        are_all_critical_points = false;
#        println("not a critical point")
#        display(Σ_r_permuted)
#        break;
    end
end
candidates
are_all_critical_points