addpath('krylov4r');
addpath('utils');

%% ------------------------------------------------------------------------
% Theorem 22: simulateneous prescription of two convergence curves, 
% one for left-prec., the other for right-prec
% -------------------------------------------------------------------------

close all

n = 10;

%% Prescriptions
% Convergence curve for right-preconditioned GMRES
r_R = zeros(n,1);
for i=1:n
    r_R(i) = 10^(-(i-1));
end
%r_R(3:5) = r_R(3);

% Convergence curve for M-GMRES
r_L = zeros(n,1);
for i=1:n
    r_L(i) = 10^(-1.5*(i-1));
end
%r_L(3:5) = r_L(3);

% Eigenvalues of the preconditioned system
lambda = ones(n,1);

%% Build system and preconditioner

% Residual decrease vector for right-preconditioned GMRES
g_R = decrease_vector(r_R);
% Residual decrease vector for M-GMRES
g_L = decrease_vector(r_L);

% Matrix T such that g = T g_tilde
T = build_T(g_R, g_L);

% Arbitrary choice of W
W = gallery('orthog', n, 4);

%M=H'H with M=(WT(WT)')^-1 => H = (WT)^-1;
H_inv = W*T; 
H = @(x) H_inv\x;

[A_hat, b] = generate_system_for_cc(r_R, lambda, W);
A = @(x) A_hat(H_inv*x);

%A_mat = A(eye(n,n));
%A = A_mat;


%% Right preconditioned
figure; axes = gca;
[~,~,~,~,absresvec] = gmres4r(A, b, 'right_prec', H, 'tol', 0);
semilogy(axes, 0:length(r_R)-1, r_R, 'LineWidth', 7, 'Marker', 'none', 'Color', [1 0 0 0.1]); % prescribed
hold(axes, 'on');
semilogy(axes, 0:length(absresvec)-1, absresvec, '-r+'); % GMRES
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');

%% Left preconditioned
[~,~,~,~,absresvec] = gmres4r(A, b, 'left_prec', H, 'tol', 0);
semilogy(axes, 0:length(r_L)-1, r_L, 'LineWidth', 7, 'Marker', 'none', 'Color', [0 0 1 0.1]); % prescribed
semilogy(axes, 0:length(absresvec)-1, absresvec, '-b+'); % GMRES
legend(axes, 'prescribed left', 'GMRES left', 'prescribed right', 'GMRES right');

