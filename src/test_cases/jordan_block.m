function A = jordan_block(n, alpha)
    if nargin < 2
        alpha = 0.99;
    end
    A = gallery('tridiag', n,  0, 1, alpha);
end


