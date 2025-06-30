addpath('krylov4r');
addpath('test_cases');

%% ------------------------------------------------------------------------
% This script implements Corollary 24.
% Let (A, b) define a given linear system, and H a preconditioner. 
% Denote by gL (resp. gR) the residual decrease vector realized by 
% I-GMRES(A, b) preconditioned by H on the left (resp. on the right).
% We construct a system (A_tilde, b_tilde) and a preconditioner H_tilde such that 
% I-GMRES(A_tilde, b_tilde) realizes gR when preconditioned by H_tilde on the left, 
% and gL when preconditioned by H_tilde on the right.
% -------------------------------------------------------------------------

close all

%% ----------------- Experiment parameters
% Matrix A
% This one is chosen because it left and right preconditioning make a
% sensible difference.
load test_cases/gm_mcfe_765.mat

% Preconditioner
H = 1/2 * (A+A');

% We set a max number of iterations for GMRES and consider that we have a breakdown, 
% otherwise it takes too long and it causes numerical issues when the 
% convergence curve goes too far. 
maxit = 100;

%% ---------------------------------------

n = size(A, 1);
b = rand(n, 1);

if isa(H, 'function_handle')
    apply_H = @(x) H(x);
elseif isempty(H)
    apply_H = @(x) x;
else
    apply_H = @(x) H\x;
end

%% ---------- GMRES(A,b) with preconditioner H
[~,~,~,~,absresvec] = gmres4r(A, b, 'left_prec', apply_H, 'maxit', maxit);

figure; axes = gca;
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'o');
title('GMRES($A,b$) with preconditioner $H$', 'Interpreter', 'latex');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');
r_L = absresvec;

[~,~,~,~,absresvec] = gmres4r(A, b, 'right_prec', apply_H, 'maxit', maxit);
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'x');
legend(axes, 'Left prec', 'Right prec');
r_R = absresvec;

%% ---------- Build (A_tilde,b_tilde) and preconditioner H_tilde

m = length(r_L)-1;

% Residual decrease vectors
g_L = zeros(n,1);
for i=1:m-1
    g_L(i) = sqrt(r_L(i)^2 - r_L(i+1)^2);
end
g_L(m) = r_L(end);

g_R = zeros(n,1);
for i=1:m-1
    g_R(i) = sqrt(r_R(i)^2 - r_R(i+1)^2);
end
g_R(m) = r_R(end);

% A_L = HA
apply_A_L = @(x) apply_H(A*x);
% b_L = Hb
b_L = apply_H(b);

% Orthonormal basis W of A_L*K(A_L,b_L)
W = zeros(n,m);
w = apply_A_L(b_L);
W(:,1) = w / norm(w);
for i=2:m
    W(:,i) = apply_A_L(W(:,i-1));
    for j=1:i-1
        W(:,i) = W(:,i) - (W(:,j)'*W(:,i)) * W(:,j);
    end
    W(:,i) = W(:,i) / norm(W(:,i));
end
% We complete the basis to have a basis of R^n
if n > m
    W = [W, null(W')]; % Im(W)+Ker(W') = R^n
end

% T
T = build_T(g_L, g_R, n);

% H_tilde = T^-1 * W'
H_tilde_inv = W*T;
apply_H_tilde = @(x) H_tilde_inv\x;

% A_tilde = A_L * H_tilde^-1
apply_A_tilde = @(x) apply_A_L(H_tilde_inv*x);

b_tilde = b_L;

%% ---------- GMRES(A_tilde,b_tilde) with preconditioner H_tilde
[~,~,~,~,absresvec] = gmres4r(apply_A_tilde, b_tilde, 'left_prec', apply_H_tilde, 'maxit', maxit);

figure; axes = gca;
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'o');
title('GMRES($\widetilde{A}$, $\widetilde{b}$) with preconditioner $\widetilde{H}$', 'Interpreter', 'latex');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

[~,~,~,~,absresvec] = gmres4r(apply_A_tilde, b_tilde, 'right_prec', apply_H_tilde, 'maxit', maxit);
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'x');
legend(axes, 'Left prec', 'Right prec');









