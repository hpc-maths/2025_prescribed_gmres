function A = laplacian(n, dirichlet_enforcement)
% LAPLACIAN: creates the 1D Laplacian matrix with Dirichlet BC arising from
% finite differences.

    if nargin < 2
        dirichlet_enforcement = 0;
    end
    
    if dirichlet_enforcement == 0 % elimination
        A = (n^2) * gallery('tridiag', n-1, -1,2,-1);
    elseif dirichlet_enforcement == 1 % boundary DoFs with 1 on the diagonal
        A = (n^2) * gallery('tridiag', n+1, -1,2,-1);
        A(1, 1:2) = [1, 0];
        A(2, 1) = 0;
        A(end, end-1:end) = [0, 1];
    else % penalization
        pen_factor = dirichlet_enforcement;
        A = (n^2) * gallery('tridiag', n+1, -1,2,-1);
        A(1, 1) = pen_factor;
        A(end, end) = pen_factor;
    end
end