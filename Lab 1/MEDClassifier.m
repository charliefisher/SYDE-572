function classes = MEDClassifier(mus, x, y)
    gridsize = size(x);
    classes = zeros(gridsize(1), gridsize(2));

    for i = 1:gridsize(1)
        for j = 1:gridsize(2)
            d = [];
            for k = 1:size(mus,2)
                d(end+1) = -mus(:,k)'*[x(i,j); y(i,j)] + 0.5*mus(:,k)'*mus(:,k);
            end
            [min_val, min_idx] = min(d);

            classes(i,j) = min_idx;
        end
    end
end