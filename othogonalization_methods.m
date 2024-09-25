addpath('gcr');
addpath('test_cases');

close all

%% Problem construction
n = 50;
DH = diag(rand(n, 1));
DH(1,1) = 10^8;
O = gallery('orthog', n);
H = O*DH*O';

D = diag(rand(n, 1)+1);
D(1,1) = 1;
%A = D*inv(H);
O2 = gallery('orthog', n, 2);
B = O2*D*O2';
A = @(x) B*(H\x);
condest(B*inv(H))

x = rand(n, 1);
b = A(x);

if isa(H, 'function_handle')
    apply_H = @(x) H(x);
else
    apply_H = @(x) H*x;
end

%% Solver parameters
tol   = 1e-12;
maxit = n;


%% -------------- GCR - Gram-Schmidt
[~,~,~,~,~,relresvec] = gcr(A, b, [], tol, maxit, apply_H, [], 'orthog_algo', 'gs');

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
ylabel(axes, 'Preconditioned residual');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% -------------- GCR - Gram-Schmidt (2)
[~,~,~,~,~,relresvec] = gcr(A, b, [], tol, maxit, apply_H, [], 'orthog_algo', 'gs', 'orthog_steps', 2);
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', '+');

%% -------------- GCR - Modified Gram-Schmidt
[~,~,~,~,~,relresvec] = gcr(A, b, [], tol, maxit, apply_H, [], 'orthog_algo', 'mgs');
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', '+');

%% -------------- GCR - Modified Gram-Schmidt (2)
[~,~,~,~,~,relresvec] = gcr(A, b, [], tol, maxit, apply_H, [], 'orthog_algo', 'mgs', 'orthog_steps', 2);
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'square');

%% -------------- GMRES

norm_b = norm(b);
norm_Hb = norm(apply_H(b));

%% GMRES - HL
[~,~,~,~,absresvec] = gmres(A, b, [], tol, maxit, apply_H, []);
semilogy(axes, 0:length(absresvec)-1, absresvec/norm_Hb, 'Marker', '*');
legend(axes, 'GCR - GS', 'GCR - GS (2x)', 'GCR - MGS', 'GCR - MGS (2x)', 'GMRES');

