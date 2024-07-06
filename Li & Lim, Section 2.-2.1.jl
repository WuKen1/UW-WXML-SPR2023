using HomotopyContinuation
using LinearAlgebra

function solution_factors(A, B, C, r)
    m, n = size(A);
    p = size(B)[2]; @assert size(B)[1] == m
    q = size(C)[1]; @assert size(C)[2] == n
    @assert m >= p && n >= q # see pg. 2 of Li and Lim
    @assert 0 <= r <= min(p,q)

    U_B, Σ_B, V_B = svd(B, full=true);
    Σ_B = diagm(Σ_B);
    Z_B = zeros(m-size(Σ_B)[1], size(Σ_B)[2]);
    isapprox(B,  U_B * [Σ_B; Z_B] * V_B')
    
    U_C, Σ_C, V_C = svd(C, full=true);
    Σ_C = diagm(Σ_C);
    Z_C = zeros(size(Σ_C)[1], n-size(Σ_C)[2]);
    isapprox(C,  U_C * [Σ_C  Z_C] * V_C')
    
    Ã = U_B' * A * V_C;
    @var X[1:p, 1:q];
    V_B' * X
    X * U_C
    X̃ = V_B' * X * U_C;
    s = rank(B)
    t = rank(C)
    S_B = Σ_B[1:s, 1:s]
    S_C = Σ_C[1:t, 1:t]
    
    X_11 = X̃[1:s,     1:t];  X_12 = X̃[1:s,     t+1:end];
    X_21 = X̃[s+1:end, 1:t];  X_22 = X̃[s+1:end, t+1:end];
    
    A_11 = Ã[1:s,     1:t];  A_12 = Ã[1:s,     t+1:end];
    A_21 = Ã[s+1:end, 1:t];  A_22 = Ã[s+1:end, t+1:end];    
    
    # Section 2.1
    U, S, V = svd(A_11);
    U_r = U[:, 1:r];
    Σ_r = diagm(S[1:r]);
    V_r = V[:, 1:r];
    
    # note that some of the matrics are already inversed/transposed
    return (V_B, [S_B^-1, U_r, Σ_r, V_r', S_C^-1], U_C, [size(X_11) size(X_12);
                                                         size(X_21) size(X_22)]);

#    X_11_opt = S_B^-1 * U_r * Σ_r * V_r' * S_C^-1; # leave this comment in for clarity

#    X_opt = V_B * [X_11_opt          zeros(size(X_12)); # leave this comment in for clarity
#               zeros(size(X_21)) zeros(size(X_22))] * U_C';
end

function closed_form_solution(A, B, C, r)
    V_B, sf, U_C, sz = solution_factors(A, B, C, r);

    X_11_opt = reduce(*, sf);
    X_opt = V_B * [X_11_opt           zeros(sz[1,2]...);
                   zeros(sz[2,1]...)  zeros(sz[2,2]...)] * U_C';
end

m = rand(4:7);
p = rand(2:m);
n = rand(5:8);
q = rand(2:n);
@assert m >= p && n >= q # see pg. 2 of Li and Lim

A = rand(-9:9, m, n);
B = rand(-9:9, m, p);
C = rand(-9:9, q, n);

X_0 = closed_form_solution(A, B, C, 1)
A - B * X_0 * C