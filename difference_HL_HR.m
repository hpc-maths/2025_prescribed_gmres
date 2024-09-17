addpath('gcr');
addpath('test_cases');

close all

%% Problem construction
n = 50;
DH = diag(rand(n, 1));
DH(1,1) = 10^3;
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

%% Solver parameters
tol   = 1e-12;
maxit = n;

if isa(A, 'function_handle')
    apply_A = @(x) A(x);
else
    apply_A = @(x) A*x;
end

if isa(H, 'function_handle')
    apply_H = @(x) H(x);
else
    apply_H = @(x) H*x;
end

%% -------------- Minimized norm

%% HL
[~,~,~,~,~,relresvec,xvecL] = gcr(apply_A, b, [], tol, maxit, apply_H, []);

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "Minimized norm");
ylabel(axes, 'Minimized norm');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[~,~,~,~,~,relresvec,xvecR] = gcr(A, b, [], tol, maxit, [], apply_H);

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');

%% Errors
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

%% -------------- Non-preconditioned residual norm

%% HL
[~,~,~,~,~,relresvec] = gcr(apply_A, b, [], tol, maxit, apply_H, [], 'res', '');

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "Residual norm");
ylabel(axes, '||b-Ax||/||b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% HR
[~,~,~,~,~,relresvec] = gcr(apply_A, b, [], tol, maxit, [], apply_H, 'res', '');

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');




%% -------------- Preconditioned residual norm

%% GCR - HL
[~,~,~,~,~,relresvec] = gcr(apply_A, b, [], tol, maxit, apply_H, [], 'res', 'l');

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, "Preconditioned residual norm");
ylabel(axes, '||H(b-Ax)||/||Hb||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR - HR
[~,~,~,~,~,relresvec] = gcr(apply_A, b, [], tol, maxit, [], apply_H, 'res', 'r');
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x');
legend(axes, 'GCR - HL', 'GCR - HR');




%% -------------- GMRES

norm_b = norm(b);
norm_Hb = norm(apply_H(b));

%% GMRES - HL
[~,~,~,~,absresvec] = gmres(apply_A, b, [], tol, maxit, apply_H, []);
figure; axes = gca; 
semilogy(axes, 0:length(absresvec)-1, absresvec/norm_Hb, 'Marker', 'o');
title(axes, "GMRES");
hold(axes, 'on');

%% GMRES - HR
[~,~,~,~,absresvec] = gmres(apply_A, b, [], tol, maxit, [], apply_H);
semilogy(axes, 0:length(absresvec)-1, absresvec/norm_Hb, 'Marker', 'x');
legend(axes, 'GMRES - HL', 'GMRES - HR');

