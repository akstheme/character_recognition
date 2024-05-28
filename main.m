%% Hand detecetion using transfer learning
%% written by Kuldeep Yadav
%% 06/02/2021

%% 
clc;
clear all;
close all;

%% Get the datastore ready
digitDatasetPath = fullfile('G:\archive\trainingSet\trainingSet');
ds = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% auds=augmentedImageDatastore([28,28,1],ds);
% ds=splitEachLabel(ds,29000);
[Train_imgs,Test_imgs]=splitEachLabel(ds,0.7);
imageSize = [28,28,1];
lebelCount=countEachLabel(ds)
num_Class=numel(categories(ds.Labels))


layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer([5 5],10)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
Train=augmentedImageDatastore(imageSize,Train_imgs);
Test=augmentedImageDatastore(imageSize,Test_imgs);

%%
options=trainingOptions('sgdm','InitialLearnRate',0.0001, ...
    'Plots','training-progress')

[hdnet,info]=trainNetwork(Train,layers,options);
save('MNIST.mat')
[testpreds,scr]=classify(hdnet,Test);
hd_actual=Test_imgs.Labels;
match_cls=nnz(hd_actual==testpreds)
frac_res=match_cls/numel(hd_actual)
plot(info.TrainingLoss)
cm=confusionchart(hd_actual,testpreds, 'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
cm.Normalization = 'row-normalized';
sortClasses(cm,'descending-diagonal');
cm.Normalization = 'absolute';
