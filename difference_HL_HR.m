addpath('krylov4r');
addpath('test_cases');

close all

%% Problem construction
n = 100;
pb = 6;
if pb == 1
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
elseif pb == 2
    alpha = 0.99;
    %A = gallery('tridiag', n, 0, alpha, 1);
    A = jordan_block(n, alpha);
    A(1,1) = 10^8;
    x = rand(n, 1);
    b = A*x;
    apply_H = @(x) diag(diag(A))\x;
    condest(A)
elseif pb == 3
    A = convdiff(ceil(sqrt(n)));
    n = size(A, 1);
    N = (A+A')/2;
    apply_H = @(x) N\x;
    x = rand(n, 1);
    b = A*x;
elseif pb == 4
    A = gallery('orthog', n);
    alpha = 1e-8;
    A(:,1) = alpha*A(:,1) + A(:,2);
    condest(A)

    D = diag(rand(n, 1));
    beta = 1e1;
    D(1,1) = beta;
    condest(D)
    
    %H = inv(A)*D;
    apply_H = @(x) A\(D*x);
    x = rand(n, 1);
    b = A*x;
elseif pb == 5
    A = gm_rand_mat_scaling_b(n);
    
    H = diag(diag(A));
    apply_H = @(x) H\x;
%     x = rand(n, 1);
%     b = apply_H(A*x);
    b = rand(n, 1);
    x = A\b;

    condest(A)
    condest(inv(H))
%     condest(inv(H)*A)
% 
%     [VHA,~] = eig(inv(H)*A);
%     [VAH,~] = eig(A*inv(H));
%     condest(VHA)
%     condest(VAH)
elseif pb == 6
    load test_cases/gm_mcfe_765.mat
    n = size(A, 1);
    H = diag(diag(A));
    apply_H = @(x) H\x;
    b = rand(n, 1);
    x = xec;
elseif pb == 62
    load test_cases/gm_mcfe_765.mat
    n = size(A, 1);
    H = diag(diag(A));
    apply_H = @(x) H\x;
    b = rand(n, 1);
    x = xec;
elseif pb == 7
    D = diag(10.^linspace(1, 6, n));
    M = diag(rand(n, 1)) + eye(n, n) + diag(rand(n-1, 1), 1) + diag(rand(n-1, 1), -1);
    %M = D*M;
    O = gallery('orthog', n);
    %O(:,1) = 1e-3*O(:,1) + O(:,n);
    %O(:,n) = O(:,n) + 1e-2*O(:,1);
    B = D*M;
    A = O*B*O';
    H = diag(diag(A));
    apply_H = @(x) M\x;
    x = rand(n, 1);
    b = A*x;
elseif pb == 8
    n = size(D1, 1);
    D = D1;%diag(linspace(0.5, 10, n));
    X = gallery('orthog', n);
    X(:,2) = 1e-5*X(:,2) + X(:,n);
    Y = gallery('orthog', n, 2);
    Y(:,1) = 1e-4*Y(:,1) + Y(:,n);
    A = X*D*inv(Y);
    H = Y*inv(X);
    %apply_H = @(x) Y*(X\x);
    apply_H = @(x) H*x;
    x = rand(n, 1);
    b = A*x;
elseif pb == 9
    alpha = 8;
    A = gallery('tridiag', n, -alpha, alpha, -1/alpha);
    H = tril(A);
    apply_H = @(x) H\x;
    x = rand(n, 1);
    b = A*x;
end

%% Solver parameters
tol   = 1e-12;
maxit = 180;
orthog_algo = 'mgs';
orthog_steps = 1;
bkdwn_tol = 1e-16;

%% -------------- Minimized residual

%% HL
[~,~,~,~,~,relresvec,xvecL] = gcr4r(A, b, [], tol, maxit, apply_H, [], 'orthog_algo', orthog_algo, 'orthog_steps', orthog_steps, 'bkdwn_tol', bkdwn_tol);

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "Minimized residual");
ylabel(axes, 'Minimized residual norm');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[~,~,~,~,~,relresvec,xvecR] = gcr4r(A, b, [], tol, maxit, [], apply_H, 'orthog_algo', orthog_algo, 'orthog_steps', orthog_steps, 'bkdwn_tol', bkdwn_tol);

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');

%% -------------- Error
eL = zeros(1, size(xvecL, 2));
for i=1:size(xvecL, 2)
    eL(1,i) = norm(x - xvecL(:, i));
end
eR = zeros(1, size(xvecR, 2));
for i=1:size(xvecR, 2)
    eR(1,i) = norm(x - xvecR(:, i));
end

figure; axes = gca;
semilogy(axes, 0:length(eL)-1, eL, 'Marker', 'o');
title(axes, "Error");
ylabel(axes, '||e||_2');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');
semilogy(axes, 0:length(eR)-1, eR, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');

% %% -------------- Non-preconditioned residual
% 
% %% HL
% [~,~,~,~,~,relresvec] = gcr4r(A, b, [], tol, maxit, apply_H, [], 'res', '', 'orthog_algo', orthog_algo, 'orthog_steps', orthog_steps, 'bkdwn_tol', bkdwn_tol);
% 
% figure; axes = gca; 
% semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
% title(axes, "(Non preconditioned) residual");
% ylabel(axes, '||b-Ax||/||b||');
% set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
% hold(axes, 'on');
% 
% %% HR
% [~,~,~,~,~,relresvec] = gcr4r(A, b, [], tol, maxit, [], apply_H, 'res', '', 'orthog_algo', orthog_algo, 'orthog_steps', orthog_steps, 'bkdwn_tol', bkdwn_tol);
% 
% semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
% legend(axes, 'GCR - HL', 'GCR - HR');
% 
% 
% %% -------------- Preconditioned residual
% 
% %% GCR - HL
% [~,~,~,~,~,relresvec] = gcr4r(A, b, [], tol, maxit, apply_H, [], 'res', 'l', 'orthog_algo', orthog_algo, 'orthog_steps', orthog_steps, 'bkdwn_tol', bkdwn_tol);
% 
% figure; axes = gca; 
% semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
% title(axes, "Preconditioned residual");
% ylabel(axes, '||H(b-Ax)||/||Hb||');
% set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
% hold(axes, 'on');
% 
% %% GCR - HR
% [~,~,~,~,~,relresvec] = gcr4r(A, b, [], tol, maxit, [], apply_H, 'res', 'r', 'orthog_algo', orthog_algo, 'orthog_steps', orthog_steps, 'bkdwn_tol', bkdwn_tol);
% semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
% legend(axes, 'GCR - HL', 'GCR - HR');


% %% -------------- GMRES (matlab)
% 
% norm_b = norm(b);
% norm_Hb = norm(apply_H(b));
% 
% %% GMRES - HL
% [~,~,~,~,absresvec] = gmres(A, b, [], tol, maxit, apply_H, []);
% figure; axes = gca; 
% semilogy(axes, 0:length(absresvec)-1, absresvec/norm_Hb, 'Marker', 'o');
% title(axes, "GMRES (matlab)");
% set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
% hold(axes, 'on');
% 
% %% GMRES - HR
% [~,~,~,~,absresvec] = gmres(A, b, [], tol, maxit, [], apply_H);
% semilogy(axes, 0:length(absresvec)-1, absresvec/norm_Hb, 'Marker', 'x');
% legend(axes, 'GMRES (matlab) - HL', 'GMRES (matlab) - HR');


%% -------------- GMRES (custom)

%% GMRES - HL
[~,~,~,~,~,relresvec] = gmres4r(A, b, [], tol, maxit, apply_H, []);
figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "GMRES (custom)");
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GMRES - HR
[~,~,~,~,~,relresvec] = gmres4r(A, b, [], tol, maxit, [], apply_H, 'orthog_algo', 'mgs', 'orthog_steps', 2);
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GMRES (custom) - HL', 'GMRES (custom) - HR');
