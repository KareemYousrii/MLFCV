% Specify the number of training points
% belonging to each of the two classes
n1 = 80; n2 = 40;

% For each class, specify the mean
% as well as the covariance
S1 = eye(2); S2 = [1 0.95; 0.95 1];
m1 = [0.75; 0]; m2 = [-0.75; 0];

% For random samples drawn from a gaussian
% N(mu, sigma^2), where mu is the mean
% and sigma^2 is the variance, we use
% sigma * gpml_randn() + mu, where gpml_randn()
% draws samples from a standard Normal distribution
x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);

% Concatenate the drawn training points into a training data
% set, and create the corresponding labels
x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';

% Generate the test dataset the same way as the training
% dataset, only changing the seed for the random number
% generator as well as the size of the test dataset
t1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.5, 2, 20), m1);
t2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.6, 2, 20), m2);

% Concatenate the generated test points into a test dataset
t = [t1 t2]';
ys = [-ones(1,20) ones(1,20)]';

% Set the mean function, covariance function, likelihood function,
% inference method as well as the hyperparameters for each function
% The cov function paratemeters consist of one characteristic length-scale
% parameter for each dimension of the input space, as well as a signal
% magnitude parameter i.e. the signal variance, for a total of 3 parameters
meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard;   hyp.cov = log([1 1 1]);
likfunc = @likErf;

% We train the hyperparameters, minimizing the negative log "marginal"
% likelihood
hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
[ymu, ys2, fmu, fs2, lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(40,1));

output = (sum((ymu<0) ~= (ys<0))/40) * 100;
disp('Percentage of classification errors:');
disp(output);

% Separate the probabilities for the misclassified
% and the correctly classifed points, taking the
% exponent in order to convert the log probabilities
% into probabilities
misclassified = exp(lp(find((ymu<0) ~= (ys<0))));
correctly_classified = exp(lp(find(~(ymu<0) ~= (ys<0))));

% calculate the entropy for the misclassified as well as the correctly
% classified samples
mc = -(misclassified .* log(misclassified) + (1 - misclassified) .* log(1-misclassified));
cc = -(correctly_classified .* log(correctly_classified) + ...
    (1 - correctly_classified) .* log(1-correctly_classified));

% Plot the histograms for the entropy.
histogram(mc, 10)
histogram(cc, 10)









