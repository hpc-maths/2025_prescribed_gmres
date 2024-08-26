addpath('gcr');

%% System to solve
A = convdiff(20, 0.01); % Convection diffusion problem
b = ones(size(A, 1), 1);

%% Solver parameters
restart = [];
tol     = 1e-10;
maxit   = 100;

% Preconditioners
[L, U] = ilu(A);
HL = L;
HR = U;
%HL = [];
%HR = [];

%% GMRES
[x,flag,relres,iter,resvec] = gmres(A, b, restart, tol, maxit, HL, HR);

norm_Hb = norm(HR\(HL\b));

figure; axes = gca; 
semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'o');
ylabel(axes, '||H(b-Ax)||/||Hb||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[x,flag,relres,iter,resvec] = gcr(A, b, restart, tol, maxit, HL, HR);

semilogy(axes, 0:length(resvec)-1, resvec/norm_Hb, 'Marker', 'x');
legend('GMRES','GCR');

