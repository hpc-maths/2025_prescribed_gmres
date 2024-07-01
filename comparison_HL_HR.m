addpath('gcr');

%% System to solve
n = 100;
%A = convdiff(100, 0.001);
A = laplacian(n);
b = rand(size(A, 1), 1);

%% Solver parameters
restart = [];
tol     = 1e-10;
maxit   = size(A, 1);

% Preconditioners
[L, U] = ilu(A);
H = L'*L;

condest(L'*L)

%% HL
[x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, H, [], [], [], [], [], 'res', '');

norm_b = norm(b);

figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'o');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, [], H, [], [], [], [], 'res', '');

semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'x');
legend('HL','HR');

