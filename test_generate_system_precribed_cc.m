addpath('krylov4r');

%% ------------------------------------------------------------------------
% This script builds a system realizing a prescribed convergence curve
% with GMRES.
% -------------------------------------------------------------------------

n = 10;

% Prescribed residual norms
r = zeros(n,1);
for i=1:n
    r(i) = 10^(-(i-1));
end
%r = linspace(1, exp(1e-10), n)';
r(3:5) = r(3);

% Prescribed eigenvalues
lambda = 1:n;

[A,b] = generate_system_for_cc(r, lambda);

%% GMRES
[~,~,~,~,absresvec] = gmres4r(A, b);%, 'tol', 0);
figure; axes = gca;
semilogy(axes, 0:length(absresvec)-2, r, 'LineWidth', 7, 'Marker', 'none', 'Color', [1 0 0 0.2]); % prescribed
hold(axes, 'on');
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'o', 'Color', 'b'); % GMRES
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
legend(axes, 'prescribed', 'GMRES');
