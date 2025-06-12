function [A] = generate_A_from_b_and_W(b, W, lambda)
    n = length(b);

    % Companion matrix
    coeffs = poly(lambda); % coefficients of the characteristic polynomial given by the eigenvalues
    alpha = -flip(coeffs(2:end)); % the leading coeff is 1, we don't take it
    C = diag(ones(n-1, 1), -1);
    C(:,end) = alpha;

    % Basis B
    B = zeros(n,n);
    B(:,1) = b;
    B(:,2:n) = W(:,1:n-1);
    
    % Operator A
    %A = B*C*inv(B);
    A = @(x) B*C*(B\x);

    %% computes eigenvalues of operator A
%     options.issym = false; % Set to true if the matrix is symmetric
%     options.tol = 1e-6;    % Tolerance for convergence
%     [eigenvectors, D] = eigs(A, n, n, 'lm', options);
%     eigenvalues = diag(D)

    %condest(A)
    %condest(B)
end