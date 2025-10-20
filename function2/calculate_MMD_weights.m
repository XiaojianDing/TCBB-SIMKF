function alphaK = calculate_MMD_weights(D_Kernels, S0_initial, delta)
% Compute kernel fusion weights based on similarity matching and MMD weighting
%
% Inputs:
%   D_Kernels     : (N x N x M) Candidate kernel matrices for one omics view
%   S0_initial    : (N x N) Initial similarity matrix (e.g., precomputed global structure)
%   delta         : (scalar) Temperature parameter controlling weight sharpness
%
% Output:
%   alphaK        : (M x 1) Normalized kernel weights

% Number of kernels
num_kernels = size(D_Kernels, 3);

% Initialize vector to store similarity match degree
DD = zeros(1, num_kernels);

% Compute similarity between each kernel and initial structure
for i = 1:num_kernels
    temp = (eps + D_Kernels(:, :, i)) .* (eps + S0_initial);
    DD(i) = mean(temp(:));  % kernel–structure matching score
end

% Compute weights based on scaled similarity distances
DD_scaled = (DD - min(DD)) / (max(DD) - min(DD) + eps);
weights = exp(-DD_scaled / delta);

% Normalize weights
alphaK = weights / sum(weights);

end
