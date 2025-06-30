function A = random_spd(n)
    A = sprandsym(n, 1e-1, 1e-4) + speye(n, n);
end


