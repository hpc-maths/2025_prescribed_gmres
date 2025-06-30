addpath('krylov4r');

hold off

%% Experiment parameters

n = 40; % system size
nsamples = 5;

% 1 ---- Matrix M (uncomment what you want)
M_eigenvalues = 'two_distinct'; % two distinct values with large gap
%M_eigenvalues = 'two_clusters'; % two clusters of values with large gap
%M_eigenvalues = 'three_distinct'; % three distinct values with large gap
%M_eigenvalues = 'evenly_spaced'; % evenly spread out in log scale

other_eigenvalue = 10^6; % the first eigenvalue is 1, this is the other
k = n/2; % multiplicity of the eigenvalue 1 in 'two_distinct'

shuffle_eigenvalues = 0;
reverse_eigenvalues = 0;

normalize_M = 1; % enforce ||b||_M = 1

% 2 ---- Right-hand side b (uncomment what you want)
%right_hand_side = 'evenly_distributed'; % evenly distributed on M's eigenvectors
%right_hand_side = 'randomly_distributed'; % randomly distributed on M's eigenvectors
%right_hand_side = 'increasing'; % increasing coefficients on M's eigenvectors
%right_hand_side = 'decreasing'; % increasing coefficients on M's eigenvectors
right_hand_side = 'ones'; % all 1 vector

% 3 ---- Basis of the Krylov residual space A*Kn(A,b)  (uncomment what you want)
res_krylov_basis = 'same_order';   % the eigenvectors of M in the same order
%res_krylov_basis = 'alternate';    % the eigenvectors of M in alternate order
%res_krylov_basis = 'random_order'; % the eigenvectors of M in random order

export_data_to_file = 0; % export for the paper


%% Prescribed eigenvalues of M

if strcmp(M_eigenvalues, 'two_distinct') % two distinct values with large gap
   mu = [ones(k,1); other_eigenvalue * ones(n-k,1)];
elseif strcmp(M_eigenvalues, 'two_clusters') % two clusters of values with large gap
    random = randi(100, n, 1);
    mu = random.*[ones(k,1); other_eigenvalue * ones(n-k,1)];
elseif strcmp(M_eigenvalues, 'three_distinct') % three distinct values with large gap
    mu = [ones(n-k1-k2,1); 10.^10*ones(k1,1); 10.^20*ones(k2,1)];
elseif strcmp(M_eigenvalues, 'evenly_spaced') % evenly spread out in log scale
   mu = (10.^linspace(0,-12,n))';
end

if shuffle_eigenvalues
    mu = mu(randperm(n));
end
if reverse_eigenvalues
    mu = flip(mu);
end


%% Running experiments...

export_data_rI = zeros(n, nsamples);
export_data_rM = zeros(n, nsamples);

for s = 1:nsamples

    % Random generation of M with mu as eigenvalues
   [Q, ~] = qr(rand(n)); % a random unitary matrix
    M = Q * diag(mu) * Q';

    if strcmp(right_hand_side, 'evenly_distributed')
        b = Q * ones(n,1);
    elseif strcmp(right_hand_side, 'randomly_distributed')
        b = Q * rand(n,1);
    elseif strcmp(right_hand_side, 'increasing')
        b = Q * (1:n)';
    elseif strcmp(right_hand_side, 'decreasing')
        b = Q * flip(1:n)';
    elseif strcmp(right_hand_side, 'ones')
        b =  ones(n,1);
    end
    b = b/norm(b);
  
    % Normalization of M, so that ||b||_M = 1
    norm_sq_b_M = b'*M*b;
    beta = 1/norm_sq_b_M;
    if normalize_M
        mu = mu/norm_sq_b_M;
        M  = M/norm_sq_b_M;
        norm_sq_b_M = 1;
    end

    % Basis of the Krylov residual space A*Kn(A,b)
    if strcmp(res_krylov_basis, 'same_order')
        W = Q;
    elseif strcmp(res_krylov_basis, 'alternate')
        W(:,1:2:end) = Q(:,1:n/2);  
        W(:,2:2:end) = Q(:,n/2+1:n);
    elseif strcmp(res_krylov_basis, 'random_order')
        W = Q(:,randperm(n)); 
    end

    % Basis of the Krylov space Kn(A,b)
    B = [b W(:,1:n-1)];

    % Residual norms of I-GMRES and M-GMRES by Ipsen's formula
    rI = zeros(n, 1);
    rM = zeros(n, 1);
    rM(1) = sqrt(norm_sq_b_M);
    rI(1) = norm(b);
    for i = 1:(n-1)
        e = [1; zeros(i,1)];
        rM(i+1) = sqrt(1/(e'*(  (B(:,1:i+1)'*(M*B(:,1:i+1)))  \e))); 
        rI(i+1) = sqrt(1/(e'*(  (B(:,1:i+1)'*(  B(:,1:i+1)))  \e))); 
    end
  
    % Plots
    semilogy(0:n-1, rI,'k-');
    hold on;
    semilogy(0:n-1, rM,'-');

    % Just to verify Ipsen's formulas
    % Method 1: the explicit formulas in the paper
%     i = 0:k;
%     I_formula = sqrt(1-x/n);
%     semilogy(0:n-1, I_formula, 'b-+');
%     I_formula1 = sqrt(1-beta*i/n);
%     semilogy(i, I_formula1, 'b-+');
%     
%     i = k+1:n-1;
%     I_formula2 = sqrt(I_formula1(end)^2 - beta*10^-6/n*(i-k));
%     semilogy(i, I_formula2, 'g-+');
    % Method 2: We actually build the system and run GMRES
%     A = generate_A_from_b_and_W(b, W, ones(n, 1));
%     [~,~,~,~,absresvec] = gmres4r(A, b);
%     semilogy(0:length(absresvec)-1, absresvec, 'Marker', 'o', 'Color', 'b');

    if nsamples == 1
        legend('I-GMRES', 'M-GMRES');
    end

    if export_data_to_file
        export_data_rI(:,s) = rI;
        export_data_rM(:,s) = rM;
    end
end

if export_data_to_file
    export_data = [export_data_rI export_data_rM];
    filename = ['ipsen__M_' M_eigenvalues '__b_' right_hand_side '__Kry_' res_krylov_basis '.txt'];
    save(filename, 'export_data', '-ascii');
end
