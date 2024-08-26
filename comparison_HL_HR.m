addpath('gcr');

%% System to solve
n = 100;
%A = convdiff(100, 0.001);
A = laplacian(n);
b = rand(size(A, 1), 1);

%% Solver parameters
restart = [];
tol     = 1e-8;
maxit   = size(A, 1);

% Preconditioners
%L = ichol(A);
%H = L'*L;
H = diag(diag(A));
% 
% condest(H)
% condest(inv(H)*A)
% condest(A*inv(H))
%H = @(x) SymGaussSeidel(A, b, x);

%% Non-preconditioned residual norm

%% HL
[x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, H, [], 'res', '');

norm_b = norm(b);

figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'o');
title(axes, "Residual norm");
ylabel(axes, '||b-Ax||/||b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, [], H, 'res', '');

semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');

%% Preconditioned residual norm
norm_Hb = norm(H\b);

%% GCR - HL
[x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, H, []);

figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'o');
title(axes, "Preconditioned residual norm");
ylabel(axes, '||H(b-Ax)||/||Hb||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR - HR
[x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, [], H);
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'x');

% %% GMRES - HL
% [x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, H, []);
% semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', '+');
% 
% %% GMRES - HR
% [x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, [], H);
% semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 's');

legend(axes, 'GCR - HL', 'GCR - HR');


function x = SymGaussSeidel(A, b, x, omega)
    if nargin < 4
        omega = 1;
    end
    D = diag(diag(A));
    L = tril(A, -1);
    U = triu(A, 1);
    x = (D + omega*L)\((-omega*U + (1-omega)*D)*x + omega*b);
    x = (D + omega*U)\((-omega*L + (1-omega)*D)*x + omega*b);
end