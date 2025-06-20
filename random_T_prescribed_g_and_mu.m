%% ------------------------------------------------------------------------
% This script implement the Illustrations section 4.1.
% 
% To reproduce the figures, change the parameters as follows:
% - Experiment 1 (Figure 1):
%     I_gmres_cc = 'stagnate'     (a)
%                  'linear_decay' (b)
%                  'irregular'    (c)
%     M_eigenvalues = 'two_distinct'
%
% - Experiment 2 (Figure 2): same, but with
%     M_eigenvalues = 'three_distinct'
%
% - Experiment 3 (Figure 3): same, but with
%     M_eigenvalues = 'evenly_spaced'
%
% - Experiment 4 (Figure 4):
%     nsamples = 11;
%     I_gmres_cc = 'linear_decay';
%     M_eigenvalues = 'two_distinct';
%     diagonal_T = 1;
%     shift_eigenvalues = 1;
%     k = n/2;
%
%     Then, in the section 'Prescribed eigenvalues of M', in the 
%       if strcmp(M_eigenvalues, 'two_distinct'),
%     uncomment
%        mu = [10.^12*ones(k,1); ones(n-k,1)]; % Figure (a)
%        mu = [ones(k,1); 10.^12*ones(n-k,1)]; % Figure (b)
%% ------------------------------------------------------------------------

hold off

%% -------------- Experiment parameters

n = 20; % system size
nsamples = 11; % number of M-GMRES convergence curves randomly generated

% 1 ---- Uncomment the convergence curve you want for I-GMRES

%I_gmres_cc = 'stagnate'; % stagnation
I_gmres_cc = 'linear_decay'; % linear decay in log scale
%I_gmres_cc = 'irregular'; % mix of linear decay and stagnation


% 2 ---- Uncomment the eigenvalues you want for M

M_eigenvalues = 'two_distinct'; % two distinct values with large gap
%M_eigenvalues = 'two_clusters'; % two clusters of values with large gap
%M_eigenvalues = 'three_distinct'; % three distinct values with large gap
%M_eigenvalues = 'evenly_spaced'; % evenly spread out in log scale
%M_eigenvalues = 'random'; % random in (0,1)

diagonal_T = 0; % force T to be diagonal

shuffle_eigenvalues = 0; % shuffle for each new sample (use with diagonal_T)
shift_eigenvalues   = 0; % circular shift for each new sample (use with diagonal_T)
reverse_eigenvalues = 0;

k = 8; % multiplicity of the large eigenvalue in 'two_distinct'

k1 = 7; % multiplicity of the 1st large eigenvalue in 'three_distinct'
k2 = 4; % multiplicity of the 2nd large eigenvalue in 'three_distinct'

normalize_residuals = 1;

export_data_to_file = 0; % export for the paper
filename = 'experiment1.txt';


%% -------------- Prescribed convergence curve for I-GMRES

% r = [r_0 ... r_{n-1}]
if strcmp(I_gmres_cc, 'stagnate')
    rI = ones(n,1);
elseif strcmp(I_gmres_cc, 'linear_decay')
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


%% -------------- Prescribed eigenvalues of M

if strcmp(M_eigenvalues, 'two_distinct') % two distinct values with large gap
    mu = [ones(n-k,1); 10.^12*ones(k,1)];
    %mu = [ones(k,1); 10.^12*ones(n-k,1)];
    %mu = [10.^12*ones(k,1); ones(n-k,1)];
elseif strcmp(M_eigenvalues, 'two_clusters') % two clusters of values with large gap
    random = randi(100, n,1);
    mu = random.*[ones(n-k,1); 10.^12*ones(k,1)];
elseif strcmp(M_eigenvalues, 'three_distinct') % three distinct values with large gap
    mu = [ones(n-k1-k2,1); 10.^10*ones(k1,1); 10.^20*ones(k2,1)];
elseif strcmp(M_eigenvalues, 'evenly_spaced') % evenly spread out in log scale
    mu = (10.^linspace(0,12,n))';
elseif strcmp(M_eigenvalues, 'random')
    mu = rand(n,1);
end

if reverse_eigenvalues
    mu = flip(mu);
end

% Singular values of T^-1
Sigma = sqrt(mu);


%% -------------- Running experiments...

export_data = zeros(n, nsamples+1);
export_data(:,1) = rI;

I_gmres_plot = semilogy(0:n-1, rI,'k-o', 'MarkerFaceColor', 'k');
hold on;

for s = 1:nsamples

    if shuffle_eigenvalues
        Sigma = Sigma(randperm(n));
    end
    if shift_eigenvalues && (s > 1 || shift_eigenvalues == -1)
        Sigma = circshift(Sigma, shift_eigenvalues*1);
    end

    % Random generation of T^-1 with Sigma as singular values
    if diagonal_T
        T_m1 = diag(Sigma);
    else
        [V,~] = qr(rand(n));
        [~,T_m1] = qr(diag(Sigma)*V);
    end
    
    % Residual decrease vector of M-GMRES
    gM = T_m1 * gI;
  
    % Residuals of M-GMRES
    rM = sqrt(flip(cumsum(flip(gM(1:end)).^2)));
    if normalize_residuals
        rM = rM/rM(1);
    end
    
    M_gmres_plot = semilogy(0:n-1, rM,'-');

    export_data(:,s+1) = rM;
end
legend([I_gmres_plot, M_gmres_plot], 'I-GMRES', 'M-GMRES');

if export_data_to_file
    save(filename, 'export_data', '-ascii');
end
