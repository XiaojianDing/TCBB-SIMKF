function H = generate_global_representation(view_features, method, weights, n_components)
    % Generate global representation matrix H
    % view_features: cell array of feature matrices (N × d_v)
    % method: 'concatenation', 'weighted_average', or 'pca_fusion'
    % weights: used only for 'weighted_average'
    % n_components: used only for 'pca_fusion'

    if nargin < 2
        error('view_features and method are required');
    end

    if strcmp(method, 'concatenation')
        H = [];
        for i = 1:length(view_features)
            H = [H, view_features{i}];
        end

    elseif strcmp(method, 'weighted_average')
        if nargin < 3 || isempty(weights)
            error('weights are required for weighted_average');
        end
        if length(weights) ~= length(view_features)
            error('Length of weights must match number of views');
        end
        [N, d] = size(view_features{1});
        H = zeros(N, d);
        for i = 1:length(view_features)
            H = H + weights(i) * view_features{i};
        end

    elseif strcmp(method, 'pca_fusion')
        if nargin < 4 || isempty(n_components)
            error('n_components is required for pca_fusion');
        end
        concatenated = [];
        for i = 1:length(view_features)
            concatenated = [concatenated, view_features{i}];
        end
        [~, score, ~] = pca(concatenated);
        H = score(:, 1:n_components);

    else
        error('Unknown method: %s', method);
    end
end
