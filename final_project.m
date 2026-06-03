clc;
clear;
close all;

%% FINAL PROJECT - PLANT DISEASE CLASSIFICATION
% Fast MATLAB Version
% Baseline CNN for Tomato Disease Classification

%% 1. DATASET PATH
mainPath = "C:\Users\2020\Downloads\PlantVillage-Dataset-master\PlantVillage-Dataset-master";

if ~isfolder(mainPath)
    error("Dataset ana klasörü bulunamadı.");
end

%% 2. LOAD DATASET
imds = imageDatastore(mainPath, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Toplam görüntü sayısı:");
disp(numel(imds.Files));

%% 3. SELECT ONLY TOMATO CLASSES
tbl = countEachLabel(imds);
allLabels = string(tbl.Label);

tomatoLabels = allLabels(contains(allLabels, "Tomato", "IgnoreCase", true));

idx = ismember(imds.Labels, categorical(tomatoLabels));
imdsTomato = subset(imds, idx);

disp("Tüm Tomato sınıfları:");
disp(countEachLabel(imdsTomato));

%% 4. USE 300 IMAGES FROM EACH CLASS
imdsTomato = splitEachLabel(imdsTomato, 300, "randomized");

disp("Kullanılan hızlandırılmış Tomato dataset:");
disp(countEachLabel(imdsTomato));

%% 5. RESET CATEGORIES
imdsTomato.Labels = removecats(imdsTomato.Labels);

numClasses = numel(categories(imdsTomato.Labels));

%% 6. TRAIN / VALIDATION / TEST SPLIT
[imdsTrain, imdsTemp] = splitEachLabel(imdsTomato, 0.70, "randomized");
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp, 0.50, "randomized");

imdsTrain.Labels = removecats(imdsTrain.Labels);
imdsValidation.Labels = removecats(imdsValidation.Labels);
imdsTest.Labels = removecats(imdsTest.Labels);

disp("Train set:");
disp(countEachLabel(imdsTrain));

disp("Validation set:");
disp(countEachLabel(imdsValidation));

disp("Test set:");
disp(countEachLabel(imdsTest));

%% 7. IMAGE SIZE
inputSize = [224 224 3];

augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augValidation = augmentedImageDatastore(inputSize, imdsValidation);
augTest = augmentedImageDatastore(inputSize, imdsTest);

%% 8. BASELINE CNN MODEL
layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 16, "Padding", "same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(3, 32, "Padding", "same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(3, 64, "Padding", "same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

%% 9. TRAINING OPTIONS
options = trainingOptions("adam", ...
    "InitialLearnRate", 0.001, ...
    "MaxEpochs", 5, ...
    "MiniBatchSize", 32, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", augValidation, ...
    "ValidationFrequency", 20, ...
    "Verbose", true);

%% 10. TRAIN NETWORK
disp("Baseline CNN eğitiliyor...");
netCNN = trainNetwork(augTrain, layers, options);

%% 11. TEST NETWORK
disp("Test işlemi yapılıyor...");
YPred = classify(netCNN, augTest);
YTest = imdsTest.Labels;

accuracy = mean(YPred == YTest);

disp("CNN Test Accuracy:");
disp(accuracy);

%% 12. CONFUSION MATRIX
figure;
cm = confusionchart(YTest, YPred);
cm.Title = "Baseline CNN Confusion Matrix";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";

saveas(gcf, "CNN_Confusion_Matrix.png");

%% 13. RESULT TABLE
Model = {'Baseline CNN'};
Accuracy = accuracy;

resultsTable = table(Model, Accuracy);

disp("Sonuç tablosu:");
disp(resultsTable);

writetable(resultsTable, "Final_Project_Results.xlsx");

%% 14. PRECISION - RECALL - F1 SCORE
classes = categories(YTest);
numClasses = numel(classes);

precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1score = zeros(numClasses,1);

for i = 1:numClasses

    className = classes{i};

    TP = sum((YPred == className) & (YTest == className));
    FP = sum((YPred == className) & (YTest ~= className));
    FN = sum((YPred ~= className) & (YTest == className));

    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);

    f1score(i) = 2 * precision(i) * recall(i) / ...
        (precision(i) + recall(i) + eps);
end

metricsTable = table(classes, precision, recall, f1score);

metricsTable.Properties.VariableNames = ...
    {'Class','Precision','Recall','F1Score'};

disp("Detaylı metrikler:");
disp(metricsTable);

writetable(metricsTable, "CNN_Metrics.xlsx");

%% 15. SAVE MODEL
save("trained_CNN_PlantDisease_FAST.mat", ...
    "netCNN", "accuracy", "metricsTable", "resultsTable");

disp("İşlem tamamlandı.");