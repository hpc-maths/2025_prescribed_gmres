function g = decrease_vector(r)
    n = length(r);

    g = zeros(n,1);
    for i=1:n-1
        g(i) = sqrt(r(i)^2 - r(i+1)^2);
    end
    g(n) = r(end);

end