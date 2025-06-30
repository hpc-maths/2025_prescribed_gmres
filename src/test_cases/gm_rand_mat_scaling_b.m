function A = gm_rand_mat_scaling_b(n)
%GM_RAND_MAT_SCALING_B badly scaled matrix

%
% Author G. Meurant
% September 2024
%

rng('default');
% A = randn(n,n);
% 
% for i = 1:n
% %  f = 10^(i-1);
%  f = 5^(i-1);
%  for j = 1:n
%   A(i,j) = f * A(i,j);
%  end % for j
% end % for i

%M = randn(n,n);
M = diag(randn(n, 1)) + diag(randn(n-1, 1), 1) + diag(randn(n-1, 1), -1);
for i = 1:n
    M(i,i) = M(i,i)+4;
end

D = diag(linspace(1, 10^6, n));
% for i = 1:n
%     D(i,i) = 5^(i-1);
% end

%A = D*M;
A = D*M;
