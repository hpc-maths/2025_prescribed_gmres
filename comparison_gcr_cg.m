addpath('gcr');

%% System to solve
n = 100;
%A = laplacian(n);
%A = 1/(n+1)*gallery('tridiag', n, -1, 2,-1); % 1D diffusion problem
A = sprandsym(n, 1e-1, 1e-4) + speye(n, n);

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
W = inv(A); % Weight matrix
[x,flag,relres,iter,resvec] = gcr(A, b, [], tol, maxit, HL, HR, 'weight', W, 'res', ''); % For comparison, we don't want the weighted norm of the residual

semilogy(axes, 0:length(resvec)-1, resvec/norm_b, 'Marker', 'x');
legend('CG','GCR');

