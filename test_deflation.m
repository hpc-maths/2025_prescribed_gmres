addpath('krylov4r');
addpath('test_cases');

close all;

%% System to solve
n = 1000;
A = jordan_block(n);
b = ones(size(A, 1), 1);

%% To compute deflation spaces
M = full(1/2*(A+A'));
N = full(1/2*(A-A'));

[VNM, DNM] = eig(N,M);

%% Solver parameters
restart = [];
tol     = 1e-10;
maxit   = n;
% Preconditioners
HL = [];
HR = [];

%% GCR
[~,~,~,~,resvec] = gcr4r(A, b, restart, tol, maxit, HL, HR);

norm_b = norm(b);

figure; axes = gca;
semilogy(axes, 0:length(resvec)-1, resvec/norm_b);
hold(axes, 'on');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');

%% Deflated GCR
defl_space_dim = [10 50 100 200 300 400 500 600];
lgd = cell(size(defl_space_dim, 2)+1, 1);
lgd{1} = strcat('m = 0');
for i = 1:size(defl_space_dim, 2)
    m = defl_space_dim(1, i);

    % Deflation space of dimension m
    Z = [real(VNM(:,1:2:m)), imag(VNM(:,1:2:m))];
    Y = A*Z;
    
    [x,flag,relres,iter,resvec] = gcr4r(A, b, restart, tol, maxit, HL, HR, 'defl', Y, Z);
    
    semilogy(axes, 0:length(resvec)-1, resvec/norm_b);
    lgd{i+1} = strcat('m = ', num2str(m));
end
legend(axes, lgd);

%% GMRES
[~,~,~,~,resvec] = gmres4r(A, b, restart, tol, maxit, HL, HR);

figure; axes = gca;
semilogy(axes, 0:length(resvec)-1, resvec/norm_b);
hold(axes, 'on');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');

%% Deflated GMRES
defl_space_dim = [10 50 100 200 300 400 500 600];
lgd = cell(size(defl_space_dim, 2)+1, 1);
lgd{1} = strcat('m = 0');
for i = 1:size(defl_space_dim, 2)
    m = defl_space_dim(1, i);

    % Deflation space of dimension m
    Z = [real(VNM(:,1:2:m)), imag(VNM(:,1:2:m))];
    Y = A*Z;
    
    [x,flag,relres,iter,resvec] = gmres4r(A, b, restart, tol, maxit, HL, HR, 'defl', Y, Z);
    
    semilogy(axes, 0:length(resvec)-1, resvec/norm_b);
    lgd{i+1} = strcat('m = ', num2str(m));
end
legend(axes, lgd);
