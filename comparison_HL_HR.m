addpath('gcr');

%% System to solve
n = 10;
%A = convdiff(100, 0.001);
A = laplacian(n);
b = rand(size(A, 1), 1);

%% Solver parameters
tol     = 1e-8;
maxit   = size(A, 1);

% Preconditioners
L = ichol(A);
H = L'*L;
%H = diag(diag(A));
%H = @(x) Jacobi(A, b, x);
%H = @(x) SymGaussSeidel(A, b, x);
%H = [];
% 
% condest(H)
% condest(inv(H)*A)
% condest(A*inv(H))
if isa(H, 'function_handle')
    apply_H = @(x) H(x);
elseif isempty(H)
    apply_H = @(x) x;
else
    apply_H = @(x) H\x;
end

%% -------------- Minimized norm

%% HL
[x,flag,relres,iter,resvec] = gcr(A, b, [], tol, maxit, apply_H, []);

norm_b = norm(b);
norm_Hb = norm(apply_H(b));

figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'o');
title(axes, "Minimized norm");
ylabel(axes, 'Minimized norm');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[x,flag,relres,iter,resvec] = gcr(A, b, [], tol, maxit, [], apply_H);

semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');

%% -------------- Non-preconditioned residual norm

%% HL
[x,flag,relres,iter,resvec] = gcr(A, b, [], tol, maxit, apply_H, [], 'res', '');

norm_b = norm(b);
norm_Hb = norm(apply_H(b));

figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'o');
title(axes, "Residual norm");
ylabel(axes, '||b-Ax||/||b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[x,flag,relres,iter,resvec] = gcr(A, b, [], tol, maxit, [], apply_H, 'res', '');

semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');




%% -------------- Preconditioned residual norm

%% GCR - HL
[x,flag,relres,iter,resvec] = gcr(A, b, [], tol, maxit, apply_H, [], 'res', 'l');

figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'o');
title(axes, "Preconditioned residual norm");
ylabel(axes, '||H(b-Ax)||/||Hb||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR - HR
[x,flag,relres,iter,resvec] = gcr(A, b, [], tol, maxit, [], apply_H, 'res', 'r');
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');




%% -------------- GMRES

%% GMRES - HL
[x,flag,relres,iter,resvec] = gmres(A, b, [], tol, maxit, apply_H, []);
figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'o');
title(axes, "GMRES");
hold(axes, 'on');

%% GMRES - HR
[x,flag,relres,iter,resvec] = gmres(A, b, [], tol, maxit, [], apply_H);
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'x');
legend(axes, 'GMRES - HL', 'GMRES - HR');



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

function x = Jacobi(A, b, x, omega)
    if nargin < 4
        omega = 1;
    end
    D = diag(diag(A));
    L = tril(A, -1);
    U = triu(A, 1);
    x = D\((-omega*(L+U) + (1-omega)*D)*x + omega*b);
end