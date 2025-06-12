hold off

%% -------------- Experiment parameters

n = 20; % system size

% 1 ---- Uncomment the convergence curve you want for I-GMRES

%I_gmres_cc = 'stagnate'; % stagnation
I_gmres_cc = 'linear_decay'; % linear decay in log scale
I_gmres_cc = 'irregular'; % mix of linear decay and stagnation

% 2 ---- Uncomment the convergence curve you want for M-GMRES

M_gmres_cc = 'stagnate'; % stagnation
%M_gmres_cc = 'linear_decay'; % linear decay in log scale
M_gmres_cc = 'stronger_linear_decay';
%M_gmres_cc = 'irregular'; % mix of linear decay and stagnation
%M_gmres_cc = 'converge_one_it'; % mix of linear decay and stagnation

export_data_to_file = 0; % export for the paper


%% -------------- Prescribed convergence curve for I-GMRES

% r = [r_0 ... r_{n-1}]
if strcmp(I_gmres_cc, 'stagnate') % stagnation
    rI = ones(n,1);
elseif strcmp(I_gmres_cc, 'linear_decay') % linear decay
    rI = zeros(n,1);
    for i=1:n
        rI(i) = 10^(-0.5*i);
    end
elseif strcmp(I_gmres_cc, 'irregular') % mix of linear decay and stagnation
    rI = zeros(n,1);
    for i=1:n
        rI(i) = 10^(-0.5*i);
    end
    rI(3:5) = rI(3);
    rI(7:9) = rI(7);
    rI(11:13) = rI(11);
    rI(15:17) = rI(15);
end
rI = rI/rI(1); % normalization

% Residual decrease vector for I-GMRES
gI = zeros(n,1);
for i=1:n-1
    gI(i) = sqrt(rI(i)^2 - rI(i+1)^2);
end
gI(n) = rI(end);


%% -------------- Prescribed convergence curve for M-GMRES

% r = [r_0 ... r_{n-1}]
if strcmp(M_gmres_cc, 'stagnate')
    rM = ones(n,1);
elseif strcmp(M_gmres_cc, 'linear_decay')
    rM = zeros(n,1);
    for i=1:n
        rM(i) = 10^(-0.5*i);
    end
elseif strcmp(M_gmres_cc, 'stronger_linear_decay')
    rM = zeros(n,1);
    for i=1:n
        rM(i) = 10^(-0.8*i);
    end
elseif strcmp(M_gmres_cc, 'irregular') % mix of linear decay and stagnation
    rM = zeros(n,1);
    for i=1:n
        rM(i) = 10^(-0.7*i);
    end
    rM(3:5) = rM(3);
    rM(7:9) = rM(7);
    rM(11:13) = rM(11);
    rM(15:17) = rM(15);
elseif strcmp(M_gmres_cc, 'converge_one_it')
    rM = 10^-12 * ones(n,1);
    rM(1) = 1;
end
rM = rM/rM(1); % normalization

% Residual decrease vector for M-GMRES
gM = zeros(n,1);
for i=1:n-1
    gM(i) = sqrt(rM(i)^2 - rM(i+1)^2);
end
gM(n) = rM(end);

%% -------------- Plots

semilogy(0:n-1, rI, 'k-o', 'MarkerFaceColor', 'k');
hold on;
semilogy(0:n-1, rM, '-o');

% Singular values of T
T = build_T(gI, gM);
Sigma = svd(T);
bound = Sigma(1)/Sigma(n);
Sigma = Sigma/Sigma(n);
semilogy(0:n-1, flip(Sigma), '-x');

% Eigenvalues of M
mu = 1./(Sigma.^2);
mu = mu/mu(1);
%mu = flip(mu);
semilogy(0:n-1, mu, '-x');


fill([0:n-1, flip(0:n-1)]', [rI / bound ; flip(rI * bound) ]', 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

legend('I-GMRES', 'M-GMRES', 'Singular values of T', 'Eigenvalues of M');
set(gca, 'XGrid', 'off', 'YGrid', 'on', 'YMinorGrid', 'off');
      

