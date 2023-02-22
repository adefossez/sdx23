function rho = corr_coef_spearman(x, y)
%% Compute Spearman correlation coefficient

[~, ~, rk1] = unique(x);
[~, ~, rk2] = unique(y);

rho = corr_coef_pearson(rk1, rk2);
