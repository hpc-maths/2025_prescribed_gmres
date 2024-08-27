function A = convdiff(n, nu, delta)
    %%Matrix from the article below:
    %%From https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/8feeea90-d360-4e14-b2db-5075f63331ff/content
    %%CONVECTION-DIFFUSION MODEL PROBLEM
    %%LIESEN AND STRAKOS

    h = 1/(n+1); % n is N in [Liesen Strakos]

    if nargin < 2
        nu = 0.01; %problem harder when smaller
    end
    if nargin < 3
        delta = 0.34;
    end

    % implement AV from equation (2.10)
    
    Mm = h/6 * gallery('tridiag', n,  1, 4, 1);
    K  = 1/h * gallery('tridiag', n, -1, 2,-1);
    C  = 1/2 * gallery('tridiag', n, -1, 0, 1);
    
    A = nu * kron(K, Mm) + kron(Mm, (nu+delta*h)*K+C);
end


