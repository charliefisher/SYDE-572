function [out1, out2] = GEDClassifier(mu, cov, X1, X2)

    if length(mu) ~= length(cov)
        error('Size of mu and cov cell arrays must be of equal length\n');
    end

    n = length(mu);
    g = zeros(size(X1,1), size(X1,2), n);  % create a grid 
    h = zeros(size(X1,1), size(X1,2));  % create class grid

    for i = 1:length(X1)
        for j = 1:length(X2)
            
            for k = 1:n
                zk = cell2mat(mu(k));
                Sk = cell2mat(cov(k));
                x = [X1(i,j) X2(i,j)]';

                g(i,j,k) = sqrt((x-zk)'*inv(Sk)*(x-zk));
            end
            
            for k = 1:n
                if min(g(i,j,:)) == g(i,j,k)
                    h(i,j) = k;
                end
            end

        end
    end

    out1 = g;
    out2 = h;
end