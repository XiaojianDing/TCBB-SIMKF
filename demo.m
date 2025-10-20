clear;
clc;
close all

% Load kernel matrix and survival labels
load('BIC_Kernels.mat');
label = table2array(Response(:, 2:end));  % [time, status]

% Kernel normalization
KH = kcenter(KH);
KH = knorm(KH);

% Parameters
C = 5;
beta_list = 0.3:0.1:0.9;
delta_list = 0.3:0.1:0.9;

% Grid search
j = 1;
for beta = beta_list
    for delta = delta_list
        [y, ~, ~, ydata] = SIMKF(KH, C, beta, delta, 20);
        group = num2cell(num2str(y));
        p = MatSurv(label(:,1), label(:,2), group, 'NoPlot', true, 'CensorLineLength', 0);
        p = max(0, min(p, 1));
        P(j) = p;
        j = j + 1;
        [minP, order] = min(P);
    end
end

