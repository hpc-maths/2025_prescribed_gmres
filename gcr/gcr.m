function [x, flag, relres, iter, resvec] = gcr(A, b, restart, tol, maxit, HL, HR, varargin)
%GCR   Generalized Conjugate Residual Method.
%   X = GCR(A,B) attempts to solve the system of linear equations A*X = B
%   for X. The N-by-N coefficient matrix A must be square and the right
%   hand side column vector B must have length N. This uses the unrestarted
%   method with MIN(N,10) total iterations.
%
%   X = GCR(AFUN,B) accepts a function handle AFUN instead of the matrix
%   A. AFUN(X) accepts a vector input X and returns the matrix-vector
%   product A*X. In all of the following syntaxes, you can replace A by
%   AFUN.
%
%   X = GCR(A,B,RESTART) restarts the method every RESTART iterations.
%   If RESTART is [] then GCR uses the unrestarted method as above.
%
%   X = GCR(A,B,RESTART,TOL) specifies the tolerance of the method. If
%   TOL is [] then GCR uses the default, 1e-6.
%
%   X = GCR(A,B,RESTART,TOL,MAXIT) specifies the maximum number of
%   iterations.
%
%   X = GCR(A,B,RESTART,TOL,MAXIT,HL) and
%   X = GCR(A,B,RESTART,TOL,MAXIT,HL,HR) use left and right 
%   preconditioners. If HL or HR is [] then the corresponding
%   preconditioner is not applied. They may be a function handle
%   returning HL\X.
%
%   X = GCR(A,B,RESTART,TOL,MAXIT,HL,HR,'weight',W) specifies the weight  
%   matrix defining the hermitian inner product. W must be hermitian 
%   positive definite. If W is [] or not specified, then GCR uses the 
%   identity matrix.
%
%   X = GCR(A,B,RESTART,TOL,MAXIT,HL,HR,'defl',Y,Z) specifies the deflation
%   spaces.
%
%   X = GCR(A,B,RESTART,TOL,MAXIT,HL,HR,'guess',X0) specifies the first 
%   initial guess. If X0 is [] or not specified, then GCR uses the default, 
%   an all zero vector.
%
%   X = GCR(A,B,RESTART,TOL,MAXIT,HL,HR,'res',OPT) specifies how the
%   residual norm is computed. The convergence cirterion also uses the 
%   same configuration to compute the norm of B for assessing the relative
%   residual norm RELRES.
%   OPT='l': HL is applied: RELRES=norm(HL\R)/norm(HL\B)
%   OPT='r': HR is applied: RELRES=norm(HR\R)/norm(HR\B)
%   OPT='w': the W-norm is used.
%   OPT='' : RELRES=norm(R)/norm(B) 
%   Option combinations are allowed.
%   The default value depends on the presence of HL, HR, W, so that the
%   residual corresponds to the one that is minimized by the algorithm.
%   Examples:
%   If HL is provided, or HL and HR, then RELRES=norm(HL\R)/norm(HL\B).
%   If HR only, then RELRES=norm(R)/norm(B).
%   If W is provided, then the W-norm is used.
%
%   [X,FLAG] = GCR(A,B,...) also returns a convergence FLAG:
%    0 GCR converged to the desired tolerance TOL within MAXIT iterations.
%    1 GCR iterated MAXIT times but did not converge.
%    2 preconditioner M was ill-conditioned.
%    3 a breakdown occured.
%
%   [X,FLAG,RELRES] = GCR(A,B,...) also returns the relative residual
%   NORM(B-A*X)/NORM(B). If FLAG is 0, then RELRES <= TOL. Note that with
%   preconditioners the preconditioned relative residual may be used.
%   See argument 'res' for details.
%
%   [X,FLAG,RELRES,ITER] = GCR(A,B,...) also returns both the iteration
%   number at which X was computed: 0 <= ITER <= MAXIT.
%
%   [X,FLAG,RELRES,ITER,RESVEC] = GCR(A,B,...) also returns a vector of
%   the (absolute) residual norms at each iteration.
%   See argument 'res' for detail on how RESVEC is computed according
%   to the presence of preconditioners and weighted norm.

    %% Argument processing

    if nargin < 3
        restart = [];
    end
    if nargin < 4
        tol = [];
    end
    if nargin < 5
        maxit = [];
    end
    if nargin < 6
        HL = [];
    end
    if nargin < 7
        HR = [];
    end
    
    % Deflation spaces
    with_deflation = 0;
    Y = [];
    Z = [];

    i = 1;
    while i < length(varargin)
        if strcmp(varargin{i}, 'defl')
            Y = varargin{i+1};
            Z = varargin{i+2};
        end
        i = i+1;
    end

    if ~isempty(Y) && ~isempty(Z)
        with_deflation = 1;
    end

    if with_deflation

        if isa(A, 'function_handle')
            error('GCR: Deflation is not implemented if A is a function. A must be a matrix.');
        end
    
        n = size(A, 1);
        if size(A, 2) ~= n
            error('CGR: A must be square.');
        end
        if isempty(Y) || isempty(Z)
            error('GCR: None or both deflation spaces Y and Z must be set.');
        end
        if ~all(size(Y) == size(Z))
            error('GCR: The deflation spaces Y and Z must have the same sizes.');
        end
        if size(Y, 1) ~= n
            error('CGR: The deflation spaces Y and Z must have the same numbers of rows as A.');
        end
        if size(Y, 2) > n
            error('CGR: The deflation spaces Y and Z must have lower or equal number of columns than A.');
        end
        if size(b, 1) ~= n
            error('CGR: The right-hand side b must have the same number of rows than A.');
        end
    end

    %% Solving

    if ~with_deflation
        % If no deflation space, then we apply the regular GCR algorithm
        [x, flag, relres, iter, resvec] = wp_gcr(A, b, restart, tol, maxit, HL, HR, varargin{:});
    else
        % Initializations
        AZ = A*Z;
        YtAZ = Y'*AZ;
        solve_YtAZ = @(b) YtAZ\b;
        
        apply_PD = @(x) x - AZ*solve_YtAZ(Y'*x);
        apply_QD = @(x) x -  Z*solve_YtAZ(Y'*A*x);
        
        % Deflated system: PD*A*x = PD*b
        apply_PDA = @(x) apply_PD(A*x);
        PDb = apply_PD(b);
    
        % Solve deflated system with GCR
        [x, flag, relres, iter, resvec] = wp_gcr(apply_PDA, PDb, restart, tol, maxit, HL, HR, varargin{:});
    
        % x = QD*x + (I-QD)x
        x = apply_QD(x) + Z*solve_YtAZ(Y'*b);
    end
end




%% ---------------------------------- %%
%     Weighted Preconditioned CGR      %
%           (no deflation)             %
%  ----------------------------------  %
function [x, flag, relres, iter, resvec] = wp_gcr(A, b, restart, tol, maxit, HL, HR, varargin)

    %% Argument processing

    if (nargin < 2)
        error(message('GCR: Not enough input arguments.'));
    end
    
    n = size(b, 1);
    if ~isa(A, 'function_handle')
        if size(A, 1) ~= size(A, 2)
            error('CGR: A must be square.');
        elseif size(A, 1) ~= n
            error('CGR: The right-hand side b must have the same number of rows than A.');
        end
    end
    if nargin < 3
        restart = [];
    end
    if nargin < 4 || isempty(tol)
        tol = 1e-6;
    end
    if nargin < 5 || isempty(maxit)
        if isempty(restart)
            maxit = min(n, 10);
        else
            maxit = min(ceil(n/restart),10);
        end
    end
    if isempty(restart)
        maxouter = 1;
        restart = maxit;
    else
        maxouter = ceil(maxit/restart);
    end

    if nargin < 6 || isempty(HL)
        HL = @(x) x;
    end
    if ~isa(HL, 'function_handle')
        if ~all(size(HL) == size(A))
            error('CGR: The left preconditioner HL must have the same size as the matrix A.');
        end
    end

    if nargin < 7 || isempty(HR)
        HR = @(x) x;
    end
    if ~isa(HR, 'function_handle')
        if ~all(size(HR) == size(A))
            error('CGR: The right preconditioner HR must have the same size as the matrix A.');
        end
    end

    if isa(A, 'function_handle')
        apply_A = @(x) A(x);
    else
        apply_A = @(x) A*x;
    end
    if isa(HL, 'function_handle')
        apply_HL = @(x) HL(x);
    else
        apply_HL = @(x) HL\x;
    end
    if isa(HR, 'function_handle')
        apply_HR = @(x) HR(x);
    else
        apply_HR = @(x) HR\x;
    end

    apply_H = @(x) apply_HR(apply_HL(x));

    W = [];
    x0 = [];

    L_prec_res   = 1;
    R_prec_res   = 0;
    weighted_res = 1;

    i = 1;
    while i < length(varargin)
        if strcmp(varargin{i}, 'weight')
            W = varargin{i+1};
        elseif strcmp(varargin{i}, 'guess')
            x0 = varargin{i+1};
        elseif strcmp(varargin{i}, 'res')
            L_prec_res   = ~isempty(strfind(varargin{i+1}, 'l'));
            R_prec_res   = ~isempty(strfind(varargin{i+1}, 'r'));
            weighted_res = ~isempty(strfind(varargin{i+1}, 'w'));
        elseif strcmp(varargin{i}, 'defl')
            i = i+1;
        else
            error('Unknown option');
        end
        i = i+2;
    end

    if isempty(W)
        herm_prod = @(x,y) y'*x;
        norm_W = @(x) norm(x);
    else
        herm_prod = @(x,y) y'*W*x;
        norm_W = @(x) sqrt(x'*W*x);
    end

    if isempty(x0)
        x0 = zeros(n, 1);
    end

    %% How to compute the norm of b
    if weighted_res
        apply_res_norm = @(x) norm_W(x);
    else
        apply_res_norm = @(x) norm(x);
    end

    if L_prec_res && ~R_prec_res
        apply_res_prec = apply_HL;
    elseif L_prec_res && R_prec_res
        apply_res_prec = apply_H;
    elseif R_prec_res
        apply_res_prec = apply_HR;
    else
        apply_res_prec = @(x) x;
    end

    compute_b_norm = @(x) apply_res_norm(apply_res_prec(x));

    %% Initializations

    norm_Hb_W = compute_b_norm(b);

    x = x0;
    if norm(x) == 0
        r = b;
    else
        r = b - apply_A(x);
    end

    % Memory allocation for the successive (absolute) residual norms ||Hr||
    resvec = zeros(maxit+1, 1);

    % Memory allocation to hold vectors
    p     = zeros(n, restart); % p_i  (research direction)
    Ap    = zeros(n, restart); % A*p_i
    HL_Ap = zeros(n, restart); % HL*A*p_i

    %% Iterations

    FLAG_CONVERGENCE = 0;
    FLAG_DIVERGENCE  = 1;
    FLAG_PREC_ISSUE  = 2;
    FLAG_BREAKDOWN   = 3;

    iter = 0;
    flag = FLAG_DIVERGENCE;

    for outer=1:maxouter

        % Residual (except on the first iteration)
        if outer > 1
            r = b - apply_A(x);
        end

        % Apply preconditioners
        HL_r = apply_HL(r);
        z = apply_HR(HL_r);
        if ~all(isfinite(z))
            warning('GCR: issue detected after applying the preconditioner.');
            flag = FLAG_PREC_ISSUE;
            break;
        end

        % Compute residual norm and check convergence
        absres = residual_norm(r, HL_r, z, apply_HR, apply_res_norm, L_prec_res, R_prec_res);
        relres = absres/norm_Hb_W;
        resvec((outer-1)*restart + 1) = absres;

        if (relres < tol)
            flag = FLAG_CONVERGENCE;
            break;
        end

        p(:,1)     = z/norm(z); % normalization to reduce the effects of round-off
        Ap(:,1)    = apply_A(p(:,1));
        HL_Ap(:,1) = apply_HL(Ap(:,1));

        for i=1:restart
    
            iter = iter+1;
    
            % Step length in the research direction
            alpha = herm_prod(HL_r, HL_Ap(:,i)) / herm_prod(HL_Ap(:,i), HL_Ap(:,i));

            if abs(alpha) < 1e-12
                warning('GCR: breakdown.');
                flag = FLAG_BREAKDOWN;
                break;
            end
    
            % Update
            x = x + alpha *  p(:,i); % solution
            r = r - alpha * Ap(:,i); % residual

            % Apply preconditioners
            HL_r = apply_HL(r);
            z = apply_HR(HL_r);
            if ~all(isfinite(z))
                warning('GCR: issue detected after applying the preconditioner.');
                flag = FLAG_PREC_ISSUE;
                break;
            end
    
            % Compute residual norm and check convergence
            absres = residual_norm(r, HL_r, z, apply_HR, apply_res_norm, L_prec_res, R_prec_res);
            relres = absres/norm_Hb_W;
            resvec((outer-1)*restart + i+1) = absres;
    
            if (relres < tol)
                flag = FLAG_CONVERGENCE;
                break;
            end
            if i == restart
                break;
            end

            % Orthogonalization of z against the p_i's
            HL_Az = apply_HL(apply_A(z));
            p(:,i+1) = z;
            for j=1:i
                beta = herm_prod(HL_Az, HL_Ap(:,j)) / herm_prod(HL_Ap(:,j), HL_Ap(:,j));
                p(:,i+1) = p(:,i+1) - beta*p(:,j);
            end
            p(:,i+1)     = p(:,i+1)/norm(p(:,i+1)); % normalization to reduce the effects of round-off
            Ap(:,i+1)    = apply_A(p(:,i+1));
            HL_Ap(:,i+1) = apply_HL(Ap(:,i+1));
        end

        if flag ~= FLAG_DIVERGENCE
            break;
        end
    end
    % Remove unused space
    resvec = resvec(1:iter+1);

    if flag == FLAG_DIVERGENCE
        warning('GCR: the given tolerance could not be reached.');
    end
end


function absres = residual_norm(r, HL_r, H_r, apply_HR, apply_res_norm, L_prec_res, R_prec_res)
    if L_prec_res && ~R_prec_res
        absres = apply_res_norm(HL_r);        % ||HL r||
    elseif L_prec_res && R_prec_res
        absres = apply_res_norm(H_r);         % ||H r||
    elseif R_prec_res
        absres = apply_res_norm(apply_HR(r)); % ||HR r||
    else
        absres = apply_res_norm(r);           % ||r||
    end
end