function [x, flag, relres, iter, absresvec, relresvec, xvec] = gmres4r(A, b, restart, tol, maxit, ML, MR, varargin)
%GMRES4R   Generalized Conjugate Residual Method.
%   X = GMRES4R(A,B) attempts to solve the system of linear equations A*X = B
%   for X. The N-by-N coefficient matrix A must be square and the right
%   hand side column vector B must have length N. This uses the unrestarted
%   method with MIN(N,10) total iterations.
%
%   X = GMRES4R(AFUN,B) accepts a function handle AFUN instead of the matrix
%   A. AFUN(X) accepts a vector input X and returns the matrix-vector
%   product A*X. In all of the following syntaxes, you can replace A by
%   AFUN.
%
%   X = GMRES4R(A,B,RESTART) restarts the method every RESTART iterations.
%   If RESTART is [] then GMRES4R uses the unrestarted method as above.
%
%   X = GMRES4R(A,B,RESTART,TOL) specifies the tolerance of the method. If
%   TOL is [] then GMRES4R uses the default, 1e-6.
%
%   X = GMRES4R(A,B,RESTART,TOL,MAXIT) specifies the maximum number of
%   iterations.
%
%   X = GMRES4R(A,B,RESTART,TOL,MAXIT,ML) and
%   X = GMRES4R(A,B,RESTART,TOL,MAXIT,ML,MR) use left and right 
%   preconditioners. If ML or MR is [] then the corresponding
%   preconditioner is not applied. They may be a function handle
%   returning ML\X.
%
%   X = GMRES4R(A,B,RESTART,TOL,MAXIT,ML,MR,'weight',W) specifies the weight  
%   matrix defining the hermitian inner product used in the algorithm,
%   computed by y'*W*x. W must be hermitian positive definite. 
%   A function can also be passed, returning how W is applied to a vector.
%   If W is [] or not specified, then GMRES4R uses the identity.
%
%   X = GMRES4R(A,B,RESTART,TOL,MAXIT,ML,MR,'defl',Y,Z) specifies the deflation
%   spaces.
%
%   X = GMRES4R(A,B,RESTART,TOL,MAXIT,ML,MR,'guess',X0) specifies the first 
%   initial guess. If X0 is [] or not specified, then GMRES4R uses the default, 
%   an all zero vector.
%
%   X = GMRES4R(A,B,RESTART,TOL,MAXIT,ML,MR,'res',OPT) specifies how the
%   residual norm is computed. The convergence cirterion also uses the 
%   same configuration to compute the norm of B for assessing the relative
%   residual norm RELRES.
%   OPT='l': ML is applied: RELRES=norm(ML\R)/norm(ML\B)
%   OPT='r': MR is applied: RELRES=norm(MR\R)/norm(MR\B)
%   OPT='w': the W-norm is used.
%   OPT='' : RELRES=norm(R)/norm(B) 
%   Option combinations are allowed.
%   The default value depends on the presence of ML, MR, W, so that the
%   residual corresponds to the one that is minimized by the algorithm.
%   Examples:
%   If ML is provided, or ML and MR, then RELRES=norm(ML\R)/norm(ML\B).
%   If MR only, then RELRES=norm(R)/norm(B).
%   If W is provided, then the W-norm is used.
%
%   [X,FLAG] = GMRES4R(A,B,...) also returns a convergence FLAG:
%    0 GMRES4R converged to the desired tolerance TOL within MAXIT iterations.
%    1 GMRES4R iterated MAXIT times but did not converge.
%    2 preconditioner ML or MR was ill-conditioned.
%
%   [X,FLAG,RELRES] = GMRES4R(A,B,...) also returns the relative residual
%   NORM(B-A*X)/NORM(B). If FLAG is 0, then RELRES <= TOL. Note that with
%   preconditioners the preconditioned relative residual may be used.
%   See argument 'res' for details.
%
%   [X,FLAG,RELRES,ITER] = GMRES4R(A,B,...) also returns both the iteration
%   number at which X was computed: 0 <= ITER <= MAXIT.
%
%   [X,FLAG,RELRES,ITER,ABSRESVEC] = GMRES4R(A,B,...) also returns a vector of
%   the absolute residual norms at each iteration.
%   See argument 'res' for detail on how ABSRESVEC is computed according
%   to the presence of preconditioners and weighted norm.
%
%   [X,FLAG,RELRES,ITER,ABSRESVEC,RELRESVEC] = GMRES4R(A,B,...) also returns a 
%   vector of the relative residual norms at each iteration, used as 
%   convergence criterion.
%   See argument 'res' for detail on how RELRESVEC is computed according
%   to the presence of preconditioners and weighted norm.
%
%   [X,FLAG,RELRES,ITER,ABSRESVEC,RELRESVEC,XVEC] = GMRES4R(A,B,...) also 
%   returns the successive approximate solutions.

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
        ML = [];
    end
    if nargin < 7
        MR = [];
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
            error('GMRES4R: Deflation is not implemented if A is a function. A must be a matrix.');
        end
    
        n = size(A, 1);
        if size(A, 2) ~= n
            error('GMRES4R: A must be square.');
        end
        if isempty(Y) || isempty(Z)
            error('GMRES4R: None or both deflation spaces Y and Z must be set.');
        end
        if ~all(size(Y) == size(Z))
            error('GMRES4R: The deflation spaces Y and Z must have the same sizes.');
        end
        if size(Y, 1) ~= n
            error('GMRES4R: The deflation spaces Y and Z must have the same numbers of rows as A.');
        end
        if size(Y, 2) > n
            error('GMRES4R: The deflation spaces Y and Z must have lower or equal number of columns than A.');
        end
        if size(b, 1) ~= n
            error('GMRES4R: The right-hand side b must have the same number of rows than A.');
        end
    end

    %% Solving

    if ~with_deflation
        % If no deflation space, then we apply the regular GMRES algorithm
        [x, flag, relres, iter, absresvec, relresvec, xvec] = wp_gmres4r(A, b, restart, tol, maxit, ML, MR, varargin{:});
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
    
        % Solve deflated system with GMRES
        [x, flag, relres, iter, absresvec, relresvec, xvec] = wp_gmres4r(apply_PDA, PDb, restart, tol, maxit, ML, MR, varargin{:});
    
        % x = QD*x + (I-QD)x
        x = apply_QD(x) + Z*solve_YtAZ(Y'*b);
    end
end




%% ------------------------------------ %%
%     Weighted Preconditioned GMRES      %
%           (no deflation)               %
%  ------------------------------------  %
function [x, flag, relres, iter, absresvec, relresvec, xvec] = wp_gmres4r(A, b, restart, tol, maxit, ML, MR, varargin)

    %% Argument processing

    if (nargin < 2)
        error(message('GMRES4R: Not enough input arguments.'));
    end
    
    n = size(b, 1);
    if ~isa(A, 'function_handle')
        if size(A, 1) ~= size(A, 2)
            error('GMRES4R: A must be square.');
        elseif size(A, 1) ~= n
            error('GMRES4R: The right-hand side b must have the same number of rows than A.');
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
            maxit = n;
        else
            maxit = ceil(n/restart);
        end
    end
    if isempty(restart)
        maxouter = 1;
        restart = maxit;
    else
        maxouter = ceil(maxit/restart);
    end

    if nargin < 6 || isempty(ML)
        ML = @(x) x;
    end
    if ~isa(ML, 'function_handle')
        if ~all(size(ML) == size(A))
            error('GMRES4R: The left preconditioner ML must have the same size as the matrix A.');
        end
    end

    if nargin < 7 || isempty(MR)
        MR = @(x) x;
    end
    if ~isa(MR, 'function_handle')
        if ~all(size(MR) == size(A))
            error('GMRES4R: The right preconditioner MR must have the same size as the matrix A.');
        end
    end

    if isa(A, 'function_handle')
        apply_A = @(x) A(x);
    else
        apply_A = @(x) A*x;
    end
    if isa(ML, 'function_handle')
        apply_ML = @(x) ML(x);
    else
        apply_ML = @(x) ML\x;
    end
    if isa(MR, 'function_handle')
        apply_MR = @(x) MR(x);
    else
        apply_MR = @(x) MR\x;
    end

    apply_M = @(x) apply_MR(apply_ML(x));

    W = [];
    x0 = [];

    L_prec_res    = 1;
    R_prec_res    = 0;
    weighted_res  = 1;
    orthog_algo   = 'mgs'; % Gram-Schmidt
    orthog_steps  = 1;

    j = 1;
    while j < length(varargin)
        if strcmp(varargin{j}, 'weight')
            W = varargin{j+1};
        elseif strcmp(varargin{j}, 'guess')
            x0 = varargin{j+1};
        elseif strcmp(varargin{j}, 'res')
            L_prec_res   = ~isempty(strfind(varargin{j+1}, 'l'));
            R_prec_res   = ~isempty(strfind(varargin{j+1}, 'r'));
            weighted_res = ~isempty(strfind(varargin{j+1}, 'w'));
        elseif strcmp(varargin{j}, 'defl')
            j = j+1;
        elseif strcmp(varargin{j}, 'orthog_algo')
            orthog_algo = varargin{j+1};
        elseif strcmp(varargin{j}, 'orthog_steps')
            orthog_steps = varargin{j+1};
        else
            error(['GMRES4R: unknown option ' varargin{j}]);
        end
        j = j+2;
    end

    if isempty(W)
        herm_prod = @(x,y) y'*x;
        norm_W = @(x) norm(x);
    elseif isa(W, 'function_handle')
        herm_prod = @(x,y) y'*W(x);
        norm_W = @(x) sqrt(x'*W(x));
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
        apply_res_prec = apply_ML;
    elseif L_prec_res && R_prec_res
        apply_res_prec = apply_M;
    elseif R_prec_res
        apply_res_prec = apply_MR;
    else
        apply_res_prec = @(x) x;
    end

    compute_b_norm = @(x) apply_res_norm(apply_res_prec(x));

    %% Initializations

    norm_Mb_W = compute_b_norm(b);


    x = x0;
    if norm(x) == 0
        r = b;
    else
        r = b - apply_A(x);
    end

    % Memory allocation for the successive residual norms
    absresvec = zeros(maxit+1, 1); % absolute
    relresvec = zeros(maxit+1, 1); % relative

    % Memory allocation for the successive solutions
    if nargout > 6
        xvec = zeros(n, maxit+1);
    end

    % Memory allocation to hold vectors
    v  = zeros(n, restart); % v_i  (Krylov space orthonormal basis)
    Av = zeros(n, restart); % ML*A*MR * v_i
    H  = zeros(restart+1, restart+1); % Hessenberg matrix

    %% Iterations

    FLAG_CONVERGENCE = 0;
    FLAG_DIVERGENCE  = 1;
    FLAG_PREC_ISSUE  = 2;

    iter = 0;
    flag = FLAG_DIVERGENCE;

    for outer=1:maxouter % if no restart, maxouter = 1

        if outer > 1
            % Compute initial residual (except on the first iteration)
            r = b - apply_A(x);

            % Empty the vectors, just to be safe (can removed for performance)
            v(:,:)  = 0;
            Av(:,:) = 0;
            H(:,:)  = 0;
        end

        % Apply left preconditioner
        z = apply_ML(r);
        if ~all(isfinite(z))
            warning('GMRES4R: issue detected after applying the preconditioner.');
            flag = FLAG_PREC_ISSUE;
            break;
        end

        norm_z = norm_W(z);
        beta = norm_z;

        % Compute residual norm and check convergence
        %absres = residual_norm(ML_r, z, apply_MR, apply_res_norm, L_prec_res, R_prec_res);
        absres = norm_z;
        relres = absres/norm_Mb_W;
        absresvec((outer-1)*restart + 1) = absres;
        relresvec((outer-1)*restart + 1) = relres;

        if nargout > 6
            xvec(:, (outer-1)*restart + 1) = x;
        end

        if (relres < tol)
            flag = FLAG_CONVERGENCE;
            break;
        end

        e1 = eye(restart+1, 1);

        v(:,1)  = z/norm_z;
        Av(:,1) = apply_ML(apply_A(apply_MR(v(:,1))));

        for j=1:restart % if no restart, restart = maxit
    
            iter = iter+1;
            
            % Arnoldi
            v(:,j+1) = Av(:,j);
            for i=1:j
                if strcmp(orthog_algo, 'gs') % Gram-Schmidt
                    H(i,j) = herm_prod(Av(:,j), v(:,i));
                elseif strcmp(orthog_algo, 'mgs') % Modified Gram-Schmidt
                    H(i,j) = herm_prod(v(:,j+1), v(:,i));
                end
                v(:,j+1) = v(:,j+1) - H(i,j)*v(:,i);
            end
            H(j+1,j) = norm_W(v(:,j+1));

            % Minimization by QR factorization
            [Q,R] = qr(H(1:j+1, 1:j));
            g = Q'*(beta * e1(1:j+1));
            y = R(1:j,:)\g(1:j); % Here, we need to discard the last row of R, which is 0

            % The (left-preconditioned) residual norm is obtained without actually computing the residual
            norm_z = abs(g(j+1)); % TODO: modify for W-norm?

            % TO BE REMOVED (just to compare...)
%             r = b-apply_A(x);
%             z = apply_ML(r);
%             disp([num2str(norm(z)) ' = ' num2str(norm_z)]);
    
            % Save residuals
            %absres = residual_norm(ML_r, z, apply_MR, apply_res_norm, L_prec_res, R_prec_res);
            absres = norm_z;
            relres = absres/norm_Mb_W;
            absresvec((outer-1)*restart + j+1) = absres;
            relresvec((outer-1)*restart + j+1) = relres;

            % Check convergence and compute the solution if needed
            if relres < tol || j == restart || nargout > 6
                x = x0 + apply_MR(v(:, 1:j)*y);
                if nargout > 6
                    xvec(:, (outer-1)*restart + j+1) = x;
                end

                if relres < tol
                    flag = FLAG_CONVERGENCE;
                    break;
                elseif j == restart
                    x0 = x;
                    break;
                end

                % TO BE REMOVED (should never go there)
                if norm(b-apply_A(x)) < tol || H(j+1,j) < tol
                    flag = FLAG_CONVERGENCE;
                    break;
                end
            end

            % Prepare next iteration
            v(:,j+1)  = v(:,j+1)/H(j+1,j);
            Av(:,j+1) = apply_ML(apply_A(apply_MR(v(:,j+1))));

            if ~all(isfinite(Av(:,j+1)))
                warning('GMRES4R: issue detected after applying the preconditioner.');
                flag = FLAG_PREC_ISSUE;
                break;
            end
            
        end

        if flag ~= FLAG_DIVERGENCE
            break;
        end
    end
    % Remove unused space
    absresvec = absresvec(1:iter+1);
    relresvec = relresvec(1:iter+1);
    if nargout > 6
        xvec = xvec(:, 1:iter+1);
    end

    if flag == FLAG_DIVERGENCE
        warning(['GMRES4R: the given tolerance could not be reached (maxit=' num2str(maxit) ').']);
    end
end


function absres = residual_norm(ML_r, H_r, apply_MR, apply_res_norm, L_prec_res, R_prec_res)
    if L_prec_res && ~R_prec_res
        absres = apply_res_norm(ML_r);        % ||ML r||
    elseif L_prec_res && R_prec_res
        absres = apply_res_norm(H_r);         % ||H r||
    elseif R_prec_res
        absres = apply_res_norm(apply_MR(r)); % ||MR r||
    else
        absres = apply_res_norm(r);           % ||r||
    end
end