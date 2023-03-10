% classifierError - report the confusion matrix and probability of error
%                   for a classifier on a set of points 
%
% INPUTS:
% X - Nx2 matrix - points to classify
% actual_label - Nx1 matrix - actual labels for points
% classifier_func - function handle - classifies Nx2 matrix of
% points
%
% OUTPUT:
% confusion - KxK matrix - the class label for each point in meshgrid
% P_e - double - the probability of error of the classifier
function [confusion, P_e] = classifierError(X, actual_label, classifier_func)
    n_test = size(X, 1);

    % check function argument preconditions
    assert(isequal(n_test, size(actual_label, 1)));
    assert(isequal(min(actual_label, [], 'all'), 1));

    n_classes = max(actual_label);
    confusion = zeros(n_classes);

    % run classifier on test set
    pred_label = classifier_func(X);
    assert(isequal(size(pred_label), size(actual_label)));

    % populate confusion matrix with results
    for i = 1:n_test
        act = actual_label(i);
        pred = pred_label(i);
        confusion(act, pred) = confusion(act, pred) + 1;
    end
    assert(isequal(sum(confusion, 'all'), n_test));

    % P_e is all incorrect classifications / N classifications
    P_e = (n_test - trace(confusion))/n_test;
end
