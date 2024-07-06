# This function generates a "solution" x0 at random, and fits a problem p0 with that
#solution by solving a linear system.
# The function "find_start_pair" should also be doing this, but ours seems to
#perform better on the r=1 examples...
function better_start_pair(F)
    n = length(variables(F))
    x0 = vec(randn(n,1) + im * randn(n,1))
    m = length(parameters(F))
    idmat = convert(Matrix{ComplexF64}, I(m))
    b = F(x0,zeros(m))
    A = hcat([F(x0,idmat[:,i])-b for i=1:m]...)
    N = nullspace(A)
    weights = randn(size(N,2),1) + im * randn(size(N,2),1)
    pHom = vec(N * weights)
    pPart = - A \ b
    p0 = pPart + pHom
    return (x0, p0)
end