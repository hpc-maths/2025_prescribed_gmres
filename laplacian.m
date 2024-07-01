function A = laplacian(n, bc_elimination)
% LAPLACIAN: creates the 1D Laplacian matrix with Dirichlet BC arising from
% finite differences.

    if nargin < 2
        bc_elimination = 1;
    end
    
    if bc_elimination
        A = (n^2) * gallery('tridiag', n-1, -1,2,-1);
    else
        A = (n^2) * gallery('tridiag', n+1, -1,2,-1);
        A(1, 1:2) = [1, 0];
        A(2, 1) = 0;
        A(end, end-1:end) = [0, 1];
    end
end