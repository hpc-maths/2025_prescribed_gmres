addpath('krylov4r');

%% ------------------------------------------------------------------------
% Theorem 6: simulateneous prescription of two convergence curves
% -------------------------------------------------------------------------

close all

n = 10;

%% Prescriptions
% Convergence curve for I-GMRES
r = zeros(n,1);
for i=1:n
    %r(i) = 10^(-(i-1));
    r(i) = 10^(-0.5*(i-1));
end
r(3:5) = r(3);

% Convergence curve for M-GMRES
r_tilde = zeros(n,1);
for i=1:n
    %r_tilde(i) = 10^(-1.5*(i-1));
    r_tilde(i) = 10^(-0.7*(i-1));
end
r_tilde(5:8) = r_tilde(5);
%r_tilde(3:5) = r_tilde(3);

% Eigenvalues of the preconditioned system
%lambda = ones(n,1);
lambda = 1:n;

%% Build system and preconditioner

% Residual decrease vector for I-GMRES
g = zeros(n,1);
for i=1:n-1
    g(i) = sqrt(r(i)^2 - r(i+1)^2);
end
g(n) = r(end);
% Residual decrease vector for M-GMRES
g_tilde = zeros(n,1);
for i=1:n-1
    g_tilde(i) = sqrt(r_tilde(i)^2 - r_tilde(i+1)^2);
end
g_tilde(n) = r_tilde(end);

% Matrix T such that g = T g_tilde
T = build_T(g, g_tilde);
%assert(norm(g-T*g_tilde) == 0);

% Arbitrary choice of W
W = gallery('orthog', n, 4);

%M = @(x) (W*T*(W*T)')\x;
M = @(x) W*(T'\(T\(W'*x)));


[A,b] = generate_system_for_cc(r, lambda, W);
assert(norm(b - W*T*g_tilde) < 1e-15);

%% GMRES
figure; axes = gca;
semilogy(axes, 0:length(r)-1, r, 'LineWidth', 7, 'Marker', 'none', 'Color', [1 0 0 0.1]); % prescribed I-GMRES
hold(axes, 'on');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
semilogy(axes, 0:length(r_tilde)-1, r_tilde, 'LineWidth', 7, 'Marker', 'none', 'Color', [0 0 1 0.1]); % prescribed M-GMRES

[~,~,~,~,absresvec] = gmres4r(A, b);
semilogy(axes, 0:length(absresvec)-1, absresvec, '-r+'); % I-GMRES

[~,~,~,~,absresvec] = gmres4r(A, b, 'weight', M, 'tol', 0);
semilogy(axes, 0:length(absresvec)-1, absresvec, '-b+'); % M-GMRES

legend(axes, 'prescribed I', 'prescribed M', 'I-GMRES', 'M-GMRES');

