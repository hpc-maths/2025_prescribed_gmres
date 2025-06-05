function T = build_T(g, g_tilde, n)
    % Builds a triangular matrix T such that g = T g_tilde

    % Matrix size
    if nargin < 3 || isempty(n)
        n = max(length(g), length(g_tilde));
    end
    % Breakdown index
    m = min(length(g), length(g_tilde));

    T = eye(n, n);
    for i=1:m
        if g(i) ~= 0 && g_tilde(i) > 0
            T(i,i) = g(i)/g_tilde(i);
        elseif g(i) == 0 && g_tilde(i) == 0
            T(i,i) = 1;
        elseif g(i) == 0 && g_tilde(i) > 0
            T(i,i) = -g(m)/g_tilde(i);
            T(i,m) = g(m)/g_tilde(m);
        else
            T(i,i) = g(i)/g_tilde(m); % it says 1 in the paper, but it could be anything, we choose this for conditioning
            T(i,m) = g(i)/g_tilde(m);
        end
    end
end