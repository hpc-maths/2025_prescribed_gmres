addpath('krylov4r');
addpath('test_cases');

%% System to solve
n = 100;
%A = laplacian(n);
A = random_spd(n);

b = rand(size(A, 1), 1);

%% Solver parameters
tol     = 1e-10;
maxit   = 100;

% Preconditioners
L = ichol(A, struct('type','ict','droptol',1e-3,'diagcomp',1));
%HL = L*L';
%HR = L*L';
HL = [];
HR = [];

norm_b = norm(b);

%% CG
[x,flag,relres,iter,resvec] = pcg(A, b, tol, maxit, HL, HR);


figure; axes = gca;
semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'o');
hold(axes, 'on');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');

%% GCR
W = @(x) A\x; % Weight matrix = A^-1
[x,flag,relres,iter,resvec] = gcr4r(A, b, HL, HR, 'tol', tol, 'maxit', maxit, 'weight', W, 'res', ''); % For comparison, we don't want the weighted norm of the residual

semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'x');
legend('CG','GCR');

