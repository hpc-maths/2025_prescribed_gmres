addpath('krylov4r');

n = 10;

% Prescribed residual norms
r = zeros(n,1);
for i=1:n
    r(i) = 10^(-(i-1));
end
%r = linspace(1, exp(1e-10), n)';
r(3:5) = r(3);

g = zeros(n,1);
for i=1:n-1
    g(i) = sqrt(r(i)^2 - r(i+1)^2);
end
g(n) = r(end);

% Orthonormal basis
W = gallery('orthog', n, 4);
%W = eye(n, n);

% Right-hand side
b = W*g;

% Matrix
C = diag(ones(n-1, 1), -1);
C(1,n) = 1/g(n);
for i=2:n
    C(i,n) = -g(i-1)/g(n);
end

B = zeros(n,n);
B(:,1) = b;
B(:,2:n) = W(:,1:n-1);

%A = B*C*inv(B);
A = @(x) B*C*(B\x);
%A = W*C*W';
%A = C;
%condest(A)
condest(B)

%% GMRES
tol = 0;
[~,~,~,~,absresvec] = gmres4r(A, b, [], tol);
figure; axes = gca;
semilogy(axes, 0:length(absresvec)-2, r, 'Marker', '+');
hold(axes, 'on');
semilogy(axes, 0:length(absresvec)-1, absresvec, 'Marker', 'o');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
legend(axes, 'prescribed', 'GMRES');
