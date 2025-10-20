function K = kcenter(K)
% kcenter - center a kernel matrix
%
% Synopsis:
%    Kc = kcenter(K);
%
% Arguments:
%    K:    Kernel matrix
%
% Return:
%    Kc: centered kernel matrix
%
% Code-Reference:
%    Shawe-Taylor, Cristianini, 'Kernel Methods for Pattern Analysis',
%    Cambridge University Press, 2004, http://www.kernel-methods.net
%
% Authors: K. Rieck, M. Kloft
%

n = size(K,2);

if ismatrix(K)
    
    D = sum(K) / n;
    E = sum(D) / n;
    J = ones(n,1) * D;
    K = K - J - J' + E * ones(n, n);
    K = 0.5 * (K + K');
    
elseif ndims(K)==3
    
    for i=1:size(K,3)
        D = sum(K(:,:,i)) / n;
        E = sum(D) / n;
        J = ones(n,1) * D;
        K(:,:,i) = K(:,:,i) - J - J' + E * ones(n, n);
        K(:,:,i) = 0.5 * (K(:,:,i) + K(:,:,i)') + 1e-12*eye(n);
    end
end
