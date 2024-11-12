addpath('krylov4r');
addpath('test_cases');

close all

%% --------------------------------------
% This script shows that the left-preconditioned GCR with H^{-1} inner-product
% is equivalent to the right-preconditioned GCR with H inner-product.
% ---------------------------------------

%% System to solve
n = 100;
%A = convdiff(n, 0.001);
A = laplacian(n);
b = rand(size(A, 1), 1);

%% Solver parameters
tol     = 1e-8;
maxit   = size(A, 1);

% Preconditioners
L = ichol(A);
H = L'*L;

if isa(H, 'function_handle')
    apply_H = @(x) H(x);
elseif isempty(H)
    apply_H = @(x) x;
else
    apply_H = @(x) H\x;
end


%% HL
W = H;
[~,~,~,~,~,relresvec] = gcr4r(A, b, [], tol, maxit, apply_H, [], 'weight', W);


figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "||H(b-Ax)||_{H^{-1}} = ||b-Ax||_H");
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
apply_W = apply_H;
[~,~,~,~,~,relresvec] = gcr4r(A, b, [], tol, maxit, [], apply_H, 'weight', apply_W);

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GCR - HL, W=H^{-1}', 'GCR - HR, W=H');



%% HL
W = H;
[~,~,~,~,~,relresvec] = gmres4r(A, b, [], tol, maxit, apply_H, [], 'weight', W);


figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "||H(b-Ax)||_{H^{-1}} = ||b-Ax||_H");
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
apply_W = apply_H;
[~,~,~,~,~,relresvec] = gmres4r(A, b, [], tol, maxit, [], apply_H, 'weight', apply_W);

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GMRES - HL, W=H^{-1}', 'GMRES - HR, W=H');