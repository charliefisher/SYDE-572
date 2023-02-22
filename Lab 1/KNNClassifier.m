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
                d{m} = sqrt(dist(:,1).^2 + dist(:,2).^2);
                d{m} = [sort(d{m}) m*ones(size(d{m}))];
            end

            de = d{1};
            for q = 2:n
                de = [de; d{q}];
            end
            
            de = sortrows(de,1);
            de; 

            h(i,j) = mode(de(1:k,2));

        end
    end

    out1 = g;
    out2 = h;



end