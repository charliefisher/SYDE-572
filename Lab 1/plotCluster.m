% plotCluster - creates a new figure and plots points in cluster
%
% INPUTS:
% clusters - cell array - list of Nx2 matrix of points in cluster
% case_num - integer - indicates what case this is (1 or 2)
function fig = plotCluster(clusters, case_num)
    % check function argument preconditions
    assert(1 <= case_num && case_num <= 2);
    assert(length(clusters) <= 3);

    % acts like hashmap from case_num to axes limits
    x_range = [
        -2 18;
        -7 25;
    ];
    y_range = [
        2 25;
        -10 25;
    ];
    % acts like hashmap of acse_num to cluster name
    case_names = {
        {'Class A', 'Class B'};
        {'Class C', 'Class D', 'Class E'};
    };

    % color to use for cluster i
    colors = ['r'; 'b'; 'g'];

    % setup figure and axes (same for every plot)
    fig = figure;
    hold on;
    xlabel('x_1');
    ylabel('x_2');
    xlim(x_range(case_num,:));
    ylim(y_range(case_num,:));

    n_clusters = length(clusters);
    % plot points in cluster
    for i = 1:n_clusters
        c = clusters{i};
        scatter(c(:,1), c(:,2), 15, colors(i), LineWidth=1.125);
    end

    % add cluster names to legend
    cluster_names = case_names(case_num);
    legend(cluster_names{:});
end
