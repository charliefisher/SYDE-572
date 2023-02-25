function [out1, out2] = KNNClassifier(clusters, k, X1, X2)

    n = length(clusters);
    g = zeros(size(X1,1), size(X1,2), n);  % create a grid 
    h = zeros(size(X1,1), size(X1,2));  % create class grid

    for i = 1:length(X1)
        for j = 1:length(X2)
            x = [X1(i,j) X2(i,j)];
            d = cell(1,3);

            for m = 1:n
                c = clusters{m};
                dist = c - (ones(size(c)) .* x);
                dist = sqrt(dist(:,1).^2 + dist(:,2).^2);
                d{m} = [dist m*ones(size(dist))];
            end

            de = d{1};
            for m = 2:n
                de = [de; d{m}];
            end
            
            de = sortrows(de,1);
            cls = de(1:k,2);
            
            h(i,j) = mode(cls);

        end
    end

    out1 = g;
    out2 = h;



end