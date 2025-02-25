addpath('krylov4r');
addpath('test_cases');

close all

% Given system
%A = convdiff(5, 0.01); % Convection diffusion problem
A = jordan_block(20, 0.7);
b = ones(size(A, 1), 1);

n = size(A,1);

% Prescribed convergence curve
r = zeros(n,1);
for i=1:n
    r(i) = 10^(-(i-1)/2);
end

g = zeros(n,1);
for i=1:n-1
    g(i) = sqrt(r(i)^2 - r(i+1)^2);
end
g(n) = r(end);

% Build orthonormal basis for the residual Krylov space (Ab, A^2b, ..., A^nb)
W = zeros(n, n);
w = A*b;
W(:,1) = w / norm(w);
for i=2:n
    W(:,i) = A*W(:,i-1);
    for j=1:i-1
        W(:,i) = W(:,i) - (W(:,j)'*W(:,i)) * W(:,j);
    end
    W(:,i) = W(:,i) / norm(W(:,i));
end

% Projection of b onto the basis W
h = W'*b;

% Weight matrix M
d = g.^2./h.^2;
M = W * diag(d) * W';

%% GMRES in M-norm
tol = 0;
[~,~,~,~,absresvec] = gmres4r(A, b, [], [], 'tol', tol);
figure; axes = gca;
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'square'); % GMRES
hold(axes, 'on');
[~,~,~,~,absresvec] = gmres4r(A, b, [], [], 'tol', tol, 'weight', M);
semilogy(axes, 0:length(absresvec)-2, r(1:length(absresvec)-1), 'Marker', '+'); % prescribed
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'o'); % GMRES in M-norm
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
legend(axes, 'GMRES', 'prescribed', 'M-GMRES');


%% P-GMRES
P = W*diag(sqrt(d));
PL = @(x) P'*x;
PR = @(x) P'\x;
tol = 0;
[~,~,~,~,absresvec] = gmres4r(A, b, [], [], 'tol', tol);
figure; axes = gca;
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'square'); % GMRES
hold(axes, 'on');
[~,~,~,~,absresvec] = gmres4r(A, b, PL, PR, 'tol', tol);
semilogy(axes, 0:length(absresvec)-2, r(1:length(absresvec)-1), 'Marker', '+'); % prescribed
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'o'); % with preconditioner
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
legend(axes, 'GMRES', 'prescribed', 'preconditioned GMRES');
