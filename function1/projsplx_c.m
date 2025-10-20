function P = projsplx_c(X)
    % Project X onto the standard (probability) simplex
    % P is the projected matrix

    [m, n] = size(X);
    P = zeros(m, n);

    for i = 1:n
        xi = X(:,i);
        xi_sorted = sort(xi, 'descend');
        cumulative_sum = cumsum(xi_sorted) - 1;
        rho = find(xi_sorted - cumulative_sum ./ (1:m)' > 0, 1, 'last');
        theta = cumulative_sum(rho) / rho;

        % Apply the projection
        P(:,i) = max(xi - theta, 0);
    end
end
