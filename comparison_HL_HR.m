addpath('krylov4r');
addpath('test_cases');

%% System to solve
n = 100;
A = convdiff(n, 0.001);
%A = laplacian(n, 2);
b = rand(size(A, 1), 1);

%% Solver parameters
tol     = 1e-8;

% Preconditioners
%L = ichol(A);
%H = L'*L;
[H,U] = ilu(A);
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
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'left_prec', apply_H, 'tol', tol);

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "Minimized norm");
ylabel(axes, 'Minimized norm');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'right_prec', apply_H, 'tol', tol);

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');

%% -------------- Non-preconditioned residual norm

%% HL
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'left_prec', apply_H, 'tol', tol, 'res', '');

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "Residual norm");
ylabel(axes, '||b-Ax||/||b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'right_prec', apply_H, 'tol', tol, 'res', '');

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');




%% -------------- Preconditioned residual norm

%% GCR - HL
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'left_prec', apply_H, 'tol', tol, 'res', 'l');

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "Preconditioned residual norm");
ylabel(axes, '||H(b-Ax)||/||Hb||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR - HR
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'right_prec', apply_H, 'tol', tol, 'res', 'r');
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');




%% -------------- GMRES

maxit = size(A, 1);
norm_b = norm(b);
norm_Hb = norm(apply_H(b));

%% GMRES - HL
[~,~,~,~,absresvec] = gmres(A, b, [], tol, maxit, apply_H, []);
figure; axes = gca; 
semilogy(axes, 0:length(absresvec)-1, absresvec/norm_Hb, 'Marker', 'o');
title(axes, "GMRES");
hold(axes, 'on');

%% GMRES - HR
[~,~,~,~,absresvec] = gmres(A, b, [], tol, maxit, [], apply_H);
semilogy(axes, 0:length(absresvec)-1, absresvec/norm_Hb, 'Marker', 'x');
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