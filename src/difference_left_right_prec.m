addpath('krylov4r');
addpath('test_cases');

close all

%% -------------- Experiment parameters

load test_cases/gm_mcfe_765.mat %on ufl data base it is specified that the kernel dim is 6 but the smallest singular value is >1000 so there is no kernel
b = rand(size(A, 1), 1);
%b = ones(size(A, 1), 1);
n = size(A,1);

%[L,U] = ilu(A);
%H = @(x) U\(L\x);

H = 1/2 * (A+A');

tol = 1e-10;

export_data_to_file = 1;
filename_prefix = 'difference_left_right_sympart2_';

%% ---------------------------------------

if isa(H, 'function_handle')
    apply_H = @(x) H(x);
elseif isempty(H)
    apply_H = @(x) x;
else
    apply_H = @(x) H\x;
end

xexact = A\b;

%% ---------- GMRES(A,b) with left-preconditioner H

% Minimized norm (preconditioned residual)
[~,~,~,~,~,relresvec,xvec] = gmres4r(A, b, 'left_prec', apply_H, 'tol', tol);

figure; axes = gca;
semilogy(axes, 0:length(relresvec)-1, relresvec, 'b-');
title('GMRES($A,b$) with preconditioner $H$', 'Interpreter', 'latex');
set(axes, 'XGrid','off', 'YGrid','on', 'YMinorGrid','off');
hold(axes, 'on');

if export_data_to_file
    save([filename_prefix 'L_min_res.txt'], 'relresvec', '-ascii');
end

% Error
errvec = zeros(length(relresvec), 1);
for i = 1:length(relresvec)
  errvec(i) = norm(xexact - xvec(:,i));
end
semilogy(axes, 0:length(errvec)-1, errvec, 'b--');

if export_data_to_file
    save([filename_prefix 'L_error.txt'], 'errvec', '-ascii');
end

% Non-preconditioned residual norm
[~,~,~,~,~,relresvec] = gmres4r(A, b, 'left_prec', apply_H, 'tol', tol, 'res', '');
semilogy(axes, 0:length(relresvec)-1, relresvec, 'b-.');

if export_data_to_file
    save([filename_prefix 'L_nonprec_res.txt'], 'relresvec', '-ascii');
end


%% ---------- GMRES(A,b) with right-preconditioner H

% Minimized norm (non-preconditioned residual)
[~,~,~,~,~,relresvec,xvec] = gmres4r(A, b, 'right_prec', apply_H, 'tol', tol);
semilogy(axes, 0:length(relresvec)-1, relresvec, 'r-');

if export_data_to_file
    save([filename_prefix 'R_min_res.txt'], 'relresvec', '-ascii');
end

% Error
errvec = zeros(length(relresvec), 1);
for i = 1:length(relresvec)
  errvec(i) = norm(xexact - xvec(:,i));
end
semilogy(axes, 0:length(errvec)-1, errvec, 'r--');

if export_data_to_file
    save([filename_prefix 'R_error.txt'], 'errvec', '-ascii');
end

% Preconditioned residual norm
[~,~,~,~,~,relresvec] = gmres4r(A, b, 'right_prec', apply_H, 'tol', tol, 'res', 'r');
semilogy(axes, 0:length(relresvec)-1, relresvec, 'r-.');

if export_data_to_file
    save([filename_prefix 'R_prec_res.txt'], 'relresvec', '-ascii');
end

legend(axes, 'L - prec. res.', 'L - error', 'L - unprec. res.', 'R - unprec. res.', 'R - error', 'R - prec. res.');



