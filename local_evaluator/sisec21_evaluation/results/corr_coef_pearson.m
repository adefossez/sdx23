function rho = corr_coef_pearson(x, y)
%% Compute Pearson correlation coefficient

rho = (x-mean(x))'*(y-mean(y)) / (eps + sqrt((x-mean(x))'*(x-mean(x)))) / (eps + sqrt((y-mean(y))'*(y-mean(y))));
