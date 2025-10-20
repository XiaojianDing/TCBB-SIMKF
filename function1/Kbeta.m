% Kbeta
%
% Computes weighted sum of kernel matrices
%
% Usage: 
%	K = Kbeta(Ks,w);
%	K = Kbeta(Ks,w,symmetric);
%
% Input:
%	Ks - Gram matrices [n x n x M]
%	w - weights [M x 1]
%	symmetric (=false) - optional argument. If 1 the Ks
%		matrices are assumed to be symmetric which results in a
%		small speedup. 
%
% Output: 
%	K - weighted kernel matrix equiv with
%		K=0;for m=1:M, K=K+ w(m)*Ks(:,:,m);end
%
% Example: 
%	
%	K = Kbeta(randn(100,100,10),rand(10,1));
%
% Peter Gehler 07/2008 pgehler@tuebingen.mpg.de

function K = Kbeta(Ks, w, symmetric)
    if nargin < 3
        symmetric = false; % 默认非对称
    end

    [n, n2, M] = size(Ks); % 获取 Ks 的维度
    if n ~= n2
        error('Kbeta: Ks should have square kernel matrices');
    end

    if length(w) ~= M
        error('Kbeta: Length of weight vector does not match number of kernels');
    end

    % 初始化加权核矩阵
    K = zeros(n, n);

    % 加权求和
    for m = 1:M
        if symmetric
            K = K + w(m) * (Ks(:,:,m) + Ks(:,:,m)') / 2; % 对称优化
        else
            K = K + w(m) * Ks(:,:,m);
        end
    end
end
