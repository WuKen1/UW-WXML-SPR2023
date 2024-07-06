# To call this function from another file, run the command
# `include("rank_1_sum_decomposition.jl)`. This should go
# smoothly if the file that you want to call this function from
# and this file itself are both in the working directory, which
# can be changed using the cd() command if necessary.

using HomotopyContinuation
using LinearAlgebra

# Returns an X ∈ ℝᵐˣʳ and Y ∈ ℝⁿˣʳ such that X*Y' == A.
# This can be interpreted as A == sum(X[:,i] * Y[:,i]' for i = 1:r),
# which is a sum of r many rank-1 matrices, hence the name of the function.

function rank_1_sum_decomposition(A)
   m, n = size(A);
   r = rank(A);

   @var X[1:m, 1:r], Y[1:n, 1:r];

   c = rand(m+1, r);
   while ~all(c[end,:] .!= 0); c = rand(m+1, r); end

   chart_equations = vcat([c[:,i]' * vcat(vec(X[:,i]), 1) for i = 1:r]);
   equality_constraint_equations = X*Y' - A;
   
   result = solve(System(vcat(vec(equality_constraint_equations),
                              chart_equations,
                              vcat([X[i,:] - rand(r) for i=1:(r-1)]...)
                             )));
   
   s = real_solutions(result); # if all is well, there should only be one solution
   x = reshape(s[1][1:m*r], m, r);
   y = reshape(s[1][m*r .+ (1:r*n)], n, r);
   return (x,y)
end