% Load train dataset
train_images = loadMNISTImages('../Data/train-images-idx3-ubyte')';
train_labels = loadMNISTLabels('../Data/train-labels-idx1-ubyte');

% Load test dataset
test_images = loadMNISTImages('../Data/t10k-images-idx3-ubyte')';
test_labels = loadMNISTLabels('../Data/t10k-labels-idx1-ubyte');

% Fit an Adaboost classifier on the first 1000 examples of the training
% dataset
X = train_images(1:1000, :);
y = train_labels(1:1000, :);

% Train an AdaBoost ensemble of 100 Decision Trees, and output the
% test classification error.
ensemble = fitensemble(X, y, 'AdaBoostM2', 100, 'Tree');
output = ['Classification error achieved using an ensemble of 100 ' ...  
    'decision trees is ', num2str(loss(ensemble, test_images, test_labels))];
disp(output);

% Train an AdaBoost ensemble of 1000 Decision Trees, and plot the
% classification error for ensembles of different sizes.
ensemble = fitensemble(X, y, 'AdaBoostM2', 1000, 'Tree');
output = ['Classification error achieved using an ensemble of 1000 ' ...  
    'decision trees is ', num2str(loss(ensemble, test_images, test_labels))];
disp(output);

% Perform a comparison of three different types of Boosting
% algorithms: AdaBoost, LPBoost and TotalBoost

% AdaBoost
ensemble = fitensemble(X, y, 'AdaBoostM2', 200, 'Tree');
output = ['Classification error achieved using an ensemble of 200 ' ...  
    'decision trees with AdaBoost is ' ... 
    num2str(loss(ensemble, test_images(1:1000, :), test_labels(1:1000, :)))];
disp(output);

% Plot AdaBoost
figure;
plot(loss(ensemble, test_images(1:1000, :), test_labels(1:1000, :), 'Mode', 'Cumulative'));
hold on;

% LPBoost
ensemble = fitensemble(X, y, 'LPBoost', 200, 'Tree');
output = ['Classification error achieved using an ensemble of 200 ' ...  
    'decision trees with LPBoost is ' ...
    num2str(loss(ensemble, test_images(1:1000, :), test_labels(1:1000, :)))];
disp(output);

% Plot LPBoost
plot(loss(ensemble, test_images(1:1000, :), test_labels(1:1000, :), 'Mode', 'Cumulative'));

% TotalBoost
ensemble = fitensemble(X, y, 'TotalBoost', 200, 'Tree');
output = ['Classification error achieved using an ensemble of 200 ' ...  
    'decision trees with TotalBoost is ' ...
    num2str(loss(ensemble, test_images(1:1000, :), test_labels(1:1000, :)))];
disp(output);

% Plot TotalBoost
plot(loss(ensemble, test_images(1:1000, :), test_labels(1:1000, :), 'Mode', 'Cumulative'));

hold off;

xlabel('Number of trees');
ylabel('Test classifiction error');
legend('AdaBoost','LPBoost', 'TotalBoost','Location','NE');

% Train a model using AdaBoost and an ensemble size of 1000 decision trees
% on the entire dataset
ensemble = fitensemble(train_images, train_labels, 'AdaBoostM2', 1000, 'Tree');
output = ['Classification error achieved using an ensemble of 1000 ' ...  
    'decision trees with AdaBoost trained on the entier dataset is ' ... 
    num2str(loss(ensemble, test_images, test_labels))];
disp(output);

% Plot AdaBoost
figure;
plot(loss(ensemble, test_images, test_labels, 'Mode', 'Cumulative'));

