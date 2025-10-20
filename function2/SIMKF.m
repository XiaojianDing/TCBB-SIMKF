function [y, S, F, ydata, alphaK, timeOurs, converge, LF] = SIMKF(KH, c, beta,delta, k)
% SIMKF: Survival-Informed Multi-Omics Kernel Fusion for Cancer Subtyping
% Inputs:
%   D_Kernels - n x n x m kernel matrices
%   c - number of clusters
%   beta - update weight (0~1)
%   delta - parameter for MMD-based weight computation
%   k - number of neighbors (optional, default 10)
% Outputs:
%   y - cluster labels
%   S - final similarity matrix
%   F - low-dimensional embedding
%   ydata - 2D embedding for visualization
%   alphaK - fixed kernel weights
%   timeOurs - elapsed time
%   converge - convergence values during iterations
%   LF - final low-dim features

if nargin == 2
    k = 10;
end

t0 = tic;
order = 2;
no_dim = c;
NITER = 30;
num = size(KH, 1);
r = -1;

% --- Step 1: Compute initial similarity matrix S0 using average kernel
distX_initial = mean(KH, 3);
[~, idx] = sort(distX_initial, 2);
A_initial = zeros(num);
di_initial = distX_initial(sub2ind(size(distX_initial), repmat((1:num)',1,k), idx(:,2:k+1)));
rr_initial = 0.5 * (k * di_initial(:,k) - sum(di_initial(:,1:k-1),2));
id_initial = idx(:,2:k+1);

temp_initial = (repmat(di_initial(:,k),1,size(di_initial,2)) - di_initial) ./ ...
    repmat((k * di_initial(:,k) - sum(di_initial(:,1:k-1),2) + eps), 1, size(di_initial,2));
a_initial = repmat((1:num)',1,size(id_initial,2));
A_initial(sub2ind(size(A_initial),a_initial(:),id_initial(:))) = temp_initial(:);
A_initial(isnan(A_initial)) = 0;

S0_initial = Network_Diffusion(A_initial + A_initial', k);
S0_initial = NE_dn(S0_initial, 'ave');
S0_initial = (S0_initial + S0_initial') / 2;

% --- Step 2: Compute MMD-based kernel weights
alphaK = calculate_MMD_weights(KH, S0_initial, delta);

% --- Initial similarity matrix
distX = Kbeta(KH, alphaK');
[distX1, idx] = sort(distX, 2);
A = zeros(num);
di = distX1(:,2:(k+2));
rr = 0.5 * (k * di(:,k+1) - sum(di(:,1:k), 2));
id = idx(:,2:k+2);
temp = (repmat(di(:,k+1),1,size(di,2)) - di) ./ ...
    repmat((k * di(:,k+1) - sum(di(:,1:k),2) + eps), 1, size(di,2));
a = repmat((1:num)', 1, size(id,2));
A(sub2ind(size(A),a(:),id(:))) = temp(:);
A(isnan(A)) = 0;

if r <= 0
    r = mean(rr);
end
lambda = max(mean(rr), 0);

S0 = max(max(distX)) - distX;
S0 = Network_Diffusion(S0, k);
S0 = NE_dn(S0, 'ave');
S = (S0 + S0') / 2;

D0 = diag(sum(S, order));
L0 = D0 - S;
[F, ~, evs] = eig1(L0, c, 0);
F = NE_dn(F, 'ave');

for iter = 1:NITER
    distf = L2_distance_1(F', F');
    
    b = idx(:,2:end);
    a = repmat((1:num)', 1, size(b,2));
    inda = sub2ind(size(A), a(:), b(:));
    ad = reshape((distX(inda) + lambda * distf(inda)) / 2 / r, num, size(b,2));
    ad = projsplx_c(-ad')';
    A(inda) = ad(:);
    A(isnan(A)) = 0;
    
    S = (1 - beta) * A + beta * S;
    S = Network_Diffusion(S, k);
    S = (S + S') / 2;
    
    D = diag(sum(S, order));
    L = D - S;
    F_old = F;
    [F, ~, ev] = eig1(L, c, 0);
    F = NE_dn(F, 'ave');
    F = (1 - beta) * F_old + beta * F;
    evs(:, iter+1) = ev;
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    converge(iter) = fn2 - fn1;
    
    if iter < 10
        if ev(end) > 1e-6
            lambda = 1.5 * lambda;
            r = r / 1.01;
        end
    else
        if converge(iter) > 1.01 * converge(iter-1)
            S = S_old;
            break;
        end
    end
    S_old = S;
    
    distX = Kbeta(KH, alphaK');
    [distX1, idx] = sort(distX, 2);
end

LF = F;
D = diag(sum(S, order));
L = D - S;
[U,~] = eig(L);

if length(no_dim) == 1
    F = tsne_p_bo(S, [], U(:,1:no_dim));
else
    F = [];
    for i = 1:length(no_dim)
        F{i} = tsne_p_bo(S, [], U(:,1:no_dim(i)));
    end
end

timeOurs = toc(t0);
[~,center] = litekmeans(LF, c, 'replicates', 200);
[~,center] = min(dist2(center, LF), [], 2);
y = litekmeans(F, c, 'Start', center);
ydata = tsne_p_bo(S);
end

% --- Kernel fusion ---
function D = Kbeta(D_Kernels, alpha)
D = zeros(size(D_Kernels,1));
for i = 1:length(alpha)
    D = D + alpha(i) * D_Kernels(:,:,i);
end
end
