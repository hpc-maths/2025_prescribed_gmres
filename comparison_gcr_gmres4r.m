addpath('krylov4r');
addpath('test_cases');

close all

%% System to solve
A = convdiff(50, 0.01); % Convection diffusion problem
b = ones(size(A, 1), 1);

%% Solver parameters
restart = [];
maxit   = size(A,1);

% Preconditioners
[L, U] = ilu(A);
HL = L;
HR = U;
%HL = [];
%HR = [];

%% --------------- No preconditioner

%% GMRES
[~,~,~,~,~,relresvec] = gmres4r(A, b, 'restart', restart);

figure; axes = gca; 
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, 'No preconditioner');
ylabel(axes, '||b-Ax||/||b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'restart', restart);

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x', 'LineStyle', '--');
legend('GMRES4R','GCR');

%% -------------- Left preconditioner

%% GMRES
[~,~,~,~,~,relresvec] = gmres4r(A, b, 'left_prec', HL, 'restart', restart);

figure; axes = gca;
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, 'Left preconditioner');
ylabel(axes, '||HL(b-Ax)||/||HL b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'left_prec', HL, 'restart', restart);

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x', 'LineStyle', '--');
legend('GMRES4R','GCR');


%% -------------- Right preconditioner

%% GMRES
[~,~,~,~,~,relresvec] = gmres4r(A, b, 'right_prec', HR, 'restart', restart);

figure; axes = gca;
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, 'Right preconditioner');
ylabel(axes, '||b-Ax||/||b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'right_prec', HR, 'restart', restart, 'res', 'r');

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x', 'LineStyle', '--');
legend('GMRES4R','GCR');


%% -------------- Left-right preconditioners

%% GMRES
[~,~,~,~,~,relresvec] = gmres4r(A, b, 'left_prec', HL, 'right_prec', HR, 'restart', restart);

figure; axes = gca;
semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'o');
title(axes, 'Left-right preconditioners');
ylabel(axes, '||HL(b-Ax)||/||HL b||');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

%% GCR
[~,~,~,~,~,relresvec] = gcr4r(A, b, 'left_prec', HL, 'right_prec', HR, 'restart', restart, 'res', 'lr');

semilogy(axes, 0:length(relresvec)-1, relresvec, 'Marker', 'x', 'LineStyle', '--');
legend('GMRES4R','GCR');
