clc
clear

% load evaluation data
load output_TAU1_FULL.mat

% define reference measure (to which we compare)
reference_measure_idx = 2;

% convert `metrics` into cell array
metrics_cell = cell(size(metrics, 1), 1);
for k = 1:numel(metrics_cell), metrics_cell{k} = strtrim(metrics(k, :));end
for k = 1:numel(metrics_cell), metrics_cell{k} = strrep(metrics_cell{k}, '<class ''metrics.', '');end
for k = 1:numel(metrics_cell), metrics_cell{k} = metrics_cell{k}(1:end-2);end
for k = 1:numel(metrics_cell), metrics_cell{k} = strrep(metrics_cell{k}, '_', '\_');end

for idx = 1:numel(metrics_cell)
    figure(idx)
    
    min_rho_pearson  = inf;
    max_rho_pearson  = -inf;
    mean_rho_pearson = 0;
    min_rho_spearman  = inf;
    max_rho_spearman  = -inf;
    mean_rho_spearman = 0;
    
    subplot(2,2,1)
    scatter(bass(idx,:), bass(reference_measure_idx,:), 'filled')
    xlabel(metrics_cell{idx})
    ylabel(metrics_cell{reference_measure_idx})
    rho_pearson  = corr_coef_pearson(bass(idx,:)', bass(reference_measure_idx,:)');
    rho_spearman = corr_coef_spearman(bass(idx,:)', bass(reference_measure_idx,:)');
    title(sprintf('bass pear: %.2f spear: %.2f', rho_pearson, rho_spearman))
    grid on
    box on
    min_rho_pearson  = min(min_rho_pearson, rho_pearson);
    max_rho_pearson  = max(max_rho_pearson, rho_pearson);
    mean_rho_pearson = mean_rho_pearson + rho_pearson;
    min_rho_spearman  = min(min_rho_spearman, rho_spearman);
    max_rho_spearman  = max(max_rho_spearman, rho_spearman);
    mean_rho_spearman = mean_rho_spearman + rho_spearman;
    
    subplot(2,2,2)
    scatter(drums(idx,:), drums(reference_measure_idx,:), 'filled')
    xlabel(metrics_cell{idx})
    ylabel(metrics_cell{reference_measure_idx})
    rho_pearson  = corr_coef_pearson(drums(idx,:)', drums(reference_measure_idx,:)');
    rho_spearman = corr_coef_spearman(drums(idx,:)', drums(reference_measure_idx,:)');
    title(sprintf('drums pear: %.2f spear: %.2f', rho_pearson, rho_spearman))
    grid on
    box on
    min_rho_pearson  = min(min_rho_pearson, rho_pearson);
    max_rho_pearson  = max(max_rho_pearson, rho_pearson);
    mean_rho_pearson = mean_rho_pearson + rho_pearson;
    min_rho_spearman  = min(min_rho_spearman, rho_spearman);
    max_rho_spearman  = max(max_rho_spearman, rho_spearman);
    mean_rho_spearman = mean_rho_spearman + rho_spearman;
    
    subplot(2,2,3)
    scatter(other(idx,:), other(reference_measure_idx,:), 'filled')
    xlabel(metrics_cell{idx})
    ylabel(metrics_cell{reference_measure_idx})
    rho_pearson  = corr_coef_pearson(other(idx,:)', other(reference_measure_idx,:)');
    rho_spearman = corr_coef_spearman(other(idx,:)', other(reference_measure_idx,:)');
    title(sprintf('other pear: %.2f spear: %.2f', rho_pearson, rho_spearman))
    grid on
    box on
    min_rho_pearson  = min(min_rho_pearson, rho_pearson);
    max_rho_pearson  = max(max_rho_pearson, rho_pearson);
    mean_rho_pearson = mean_rho_pearson + rho_pearson;
    min_rho_spearman  = min(min_rho_spearman, rho_spearman);
    max_rho_spearman  = max(max_rho_spearman, rho_spearman);
    mean_rho_spearman = mean_rho_spearman + rho_spearman;
    
    subplot(2,2,4)
    scatter(vocals(idx,:), vocals(reference_measure_idx,:), 'filled')
    xlabel(metrics_cell{idx})
    ylabel(metrics_cell{reference_measure_idx})
    rho_pearson  = corr_coef_pearson(vocals(idx,:)', vocals(reference_measure_idx,:)');
    rho_spearman = corr_coef_spearman(vocals(idx,:)', vocals(reference_measure_idx,:)');
    title(sprintf('vocals pear: %.2f spear: %.2f', rho_pearson, rho_spearman))
    grid on
    box on
    min_rho_pearson  = min(min_rho_pearson, rho_pearson);
    max_rho_pearson  = max(max_rho_pearson, rho_pearson);
    mean_rho_pearson = mean_rho_pearson + rho_pearson;
    min_rho_spearman  = min(min_rho_spearman, rho_spearman);
    max_rho_spearman  = max(max_rho_spearman, rho_spearman);
    mean_rho_spearman = mean_rho_spearman + rho_spearman;
    
    mean_rho_pearson = mean_rho_pearson / 4;
    mean_rho_spearman = mean_rho_spearman / 4;
    fprintf('%s\t%f\t%f\t%f\t%f\t%f\t%f\n', metrics_cell{idx}, ...
        min_rho_pearson, mean_rho_pearson, max_rho_pearson, ...
        min_rho_spearman, mean_rho_spearman, max_rho_spearman);
    
    print('-dpng', '-r300', sprintf('fig%d.png', idx));
end

fprintf('Mean scores: \n')
disp([mean(bass, 2) mean(drums, 2) mean(other, 2) mean(vocals, 2)])

fprintf('Median scores: \n')
disp([median(bass, 2) median(drums, 2) median(other, 2) median(vocals, 2)])