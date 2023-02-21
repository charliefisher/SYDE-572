function plotEllipse(mu, S, c)
    [eigVec, eigVal] = eig(S);

    theta = atan2(eigVec(1,2), eigVec(1,1));
    a = sqrt(eigVal(1,1));
    b = sqrt(eigVal(2,2));

    plot_ellipse(mu(1), mu(2), theta, a, b, c)
end