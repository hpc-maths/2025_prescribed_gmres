addpath('krylov4r');
addpath('test_cases');

%% ------------------------------------------------------------------------
% This script implements, for a given system (A,b) and a prescribed convergence curve
% - a weight matrix with which weighted GMRES realizes that convervence curve (Th. 4)
% - a split preconditioner with which GMRES realizes that convergence curve.
% -------------------------------------------------------------------------

close all

%% Given system
A = convdiff(5, 0.01); % Convection diffusion problem
%A = jordan_block(20, 0.7);
b = ones(size(A, 1), 1);

n = size(A,1);
m = 15; % breakdown index

%% Prescribed convergence curve for M-GMRES and prec. GMRES
r = zeros(m,1);
for i=1:m
    r(i) = 10^(-(i-1)/2);
end
r(8:11) = r(8); % stagnation during 3 iterations

% Associated residual decrease vector
g_tilde = zeros(m,1);
for i=1:m-1
    g_tilde(i) = sqrt(r(i)^2 - r(i+1)^2);
end
g_tilde(m) = r(end);

% Orthonormal basis for the residual Krylov space (Ab, A^2b, ..., A^nb)
W = zeros(n,m);
w = A*b;
W(:,1) = w / norm(w);
for i=2:m
    W(:,i) = A*W(:,i-1);
    for j=1:i-1
        W(:,i) = W(:,i) - (W(:,j)'*W(:,i)) * W(:,j);
    end
    %norm(W(:,i))
    W(:,i) = W(:,i) / norm(W(:,i));
end
% We complete the basis to have a basis of R^n 
W = [W, null(W')]; % Im(W)+Ker(W') = R^n 

% Projection of b onto the basis W
% (If all g(i)>= 0, then g is the residual decrease vector of I-GMRES)
g = W'*b;

% Matrix T such that g = T g_tilde
T = build_T(g, g_tilde, n);

% Weight matrix M = (WT(WT)')^(-1)
P = W*T;
M = @(x) (P * P')\x;

%% GMRES in M-norm
tol = 1e-12;
[~,~,~,~,absresvec] = gmres4r(A, b, 'tol', tol);
figure; axes = gca;
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'square'); % I-GMRES
hold(axes, 'on');
[~,~,~,~,absresvec] = gmres4r(A, b, 'weight', M, 'tol', tol);
semilogy(axes, 0:length(absresvec)-2, r(1:length(absresvec)-1), 'LineWidth', 7, 'Marker', 'none', 'Color',[1 0 0 0.2]); % prescribed
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'o', 'Color', 'b'); % GMRES in M-norm
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
legend(axes, 'I-GMRES', 'prescribed conv. curve', 'M-GMRES');


%% Preconditioned GMRES
PL = @(x) P\x; % (WT)^(-1)
PR = @(x) P*x; %  WT
[~,~,~,~,absresvec] = gmres4r(A, b, 'tol', tol);
figure; axes = gca;
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'square'); % GMRES
hold(axes, 'on');
[~,~,~,~,absresvec] = gmres4r(A, b, 'left_prec', PL, 'right_prec', PR, 'tol', tol);
semilogy(axes, 0:length(absresvec)-2, r(1:length(absresvec)-1), 'LineWidth', 7, 'Marker', 'none', 'Color',[1 0 0 0.2]); % prescribed
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'o', 'Color', 'b'); % with preconditioner
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
legend(axes, 'GMRES', 'prescribed conv. curve', 'preconditioned GMRES');
