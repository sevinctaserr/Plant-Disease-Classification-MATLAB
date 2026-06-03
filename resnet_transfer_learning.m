clc;
clear;
close all;

%% RESNET-18 TRANSFER LEARNING
% Plant Disease Classification

%% 1. DATASET PATH
mainPath = "C:\Users\2020\Downloads\PlantVillage-Dataset-master\PlantVillage-Dataset-master";

%% 2. LOAD DATASET
imds = imageDatastore(mainPath, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

%% 3. SELECT TOMATO CLASSES
tbl = countEachLabel(imds);
allLabels = string(tbl.Label);

tomatoLabels = allLabels(contains(allLabels, "Tomato", "IgnoreCase", true));

idx = ismember(imds.Labels, categorical(tomatoLabels));

imdsTomato = subset(imds, idx);

%% 4. FAST DATASET
imdsTomato = splitEachLabel(imdsTomato, 300, "randomized");

imdsTomato.Labels = removecats(imdsTomato.Labels);

numClasses = numel(categories(imdsTomato.Labels));

%% 5. TRAIN / VALIDATION / TEST
[imdsTrain, imdsTemp] = splitEachLabel(imdsTomato, 0.70, "randomized");
[imdsValidation, imdsTest] = splitEachLabel(imdsTemp, 0.50, "randomized");

%% 6. IMAGE SIZE
inputSize = [224 224 3];

augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augValidation = augmentedImageDatastore(inputSize, imdsValidation);
augTest = augmentedImageDatastore(inputSize, imdsTest);

%% 7. LOAD RESNET18
net = resnet18;

lgraph = layerGraph(net);

%% 8. REPLACE FINAL LAYERS
newFc = fullyConnectedLayer(numClasses, ...
    "Name","new_fc", ...
    "WeightLearnRateFactor",10, ...
    "BiasLearnRateFactor",10);

newClassLayer = classificationLayer("Name","new_classoutput");

lgraph = replaceLayer(lgraph,"fc1000",newFc);

lgraph = replaceLayer(lgraph, ...
    "ClassificationLayer_predictions", ...
    newClassLayer);

%% 9. TRAINING OPTIONS
options = trainingOptions("adam", ...
    "MiniBatchSize",32, ...
    "MaxEpochs",5, ...
    "InitialLearnRate",1e-4, ...
    "Shuffle","every-epoch", ...
    "ValidationData",augValidation, ...
    "ValidationFrequency",20, ...
    "Verbose",true);

%% 10. TRAIN NETWORK
disp("ResNet-18 eğitiliyor...");

netTransfer = trainNetwork(augTrain,lgraph,options);

%% 11. TEST
YPred = classify(netTransfer, augTest);

YTest = imdsTest.Labels;

accuracy = mean(YPred == YTest);

disp("ResNet-18 Accuracy:");
disp(accuracy);

%% 12. CONFUSION MATRIX
figure;

cm = confusionchart(YTest, YPred);

cm.Title = "ResNet-18 Confusion Matrix";

cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";

saveas(gcf,"ResNet18_ConfusionMatrix.png");

%% 13. RESULT TABLE
Model = {'ResNet-18'};
Accuracy = accuracy;

resultsTable = table(Model, Accuracy);

disp(resultsTable);

writetable(resultsTable, "ResNet18_Results.xlsx");

%% 14. SAVE MODEL
save("trained_ResNet18.mat", ...
    "netTransfer", ...
    "accuracy");

disp("Transfer learning tamamlandı.");