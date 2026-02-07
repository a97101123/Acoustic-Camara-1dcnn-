function tw_train_1dcnn(cfg)
% 讀 train_ready2/*_tr.mat，訓練並輸出 confusion matrix png/csv
inDir = cfg.paths.trainReady;
% function tw_train_1dcnn(cfg, inDir)
% if nargin < 2 || inDir==""
%     inDir = cfg.paths.trainReady;
% end


files = dir(fullfile(inDir, "*_tr.mat"));
if isempty(files), error("No *_tr.mat in %s", inDir); end

confDir = fullfile(inDir, "confusion");
if ~isfolder(confDir), mkdir(confDir); end

for i = 1:numel(files)
    f = fullfile(files(i).folder, files(i).name);
    S = load(f, "Xtrain","Ttrain","Xval","Tval");
    XTrain = S.Xtrain; TTrain = S.Ttrain;
    XVal   = S.Xval;   TVal   = S.Tval;

    if isempty(XTrain) || isempty(XVal)
        warning("Empty data: %s", f); 
        continue;
    end

    numFeatures = size(XTrain{1},1);
    classes = categories(TTrain);
    numClasses = numel(classes);

    layers = [
        sequenceInputLayer(numFeatures)
        convolution1dLayer(3, 32, Padding="causal")
        reluLayer
        layerNormalizationLayer
        convolution1dLayer(3, 64, Padding="causal")
        reluLayer
        layerNormalizationLayer
        globalAveragePooling1dLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];

    options = trainingOptions("adam", ...
        MiniBatchSize=32, ...
        MaxEpochs=15, ...
        SequencePaddingDirection="left", ...
        ValidationData={XVal,TVal}, ...
        Verbose=false, ...
        Plots="none");

    net = trainNetwork(XTrain, TTrain, layers, options);
    [~, name] = fileparts(f);   % f 是 *_tr.mat 的完整路徑
    name = string(name);        % 保證後面 name + "_net.mat" 不會出型別問題
    modelDir = fullfile(inDir, "models");
    if ~isfolder(modelDir), mkdir(modelDir); end

    % 把訓練用的 cfg、類別順序也一起存（推論時要一致）
    save(fullfile(modelDir, name + "_net.mat"), "net", "classes", "cfg", "-v7.3");

    YPred = classify(net, XVal, MiniBatchSize=32, SequencePaddingDirection="left");
    acc = mean(YPred == TVal);
    fprintf("%s: valAcc=%.4f\n", files(i).name, acc);

    % --- 固定類別順序（用 confusionmat 的 Order 支援度比較高） ---
    ord = categorical(classes);  % classes = categories(TTrain)
    cm  = confusionmat(TVal, YPred, "Order", ord);

    % csv
    cmTable = array2table(cm, ...
        'VariableNames', cellstr("pred_" + string(classes)), ...
        'RowNames',      cellstr("true_" + string(classes)));
    [~,name] = fileparts(f);
    writetable(cmTable, fullfile(confDir, name + "_confusion.csv"), "WriteRowNames", true);

    % --- confusionchart：用 (cm, labels) 這種語法，避免 'Order' ---
    labels = cellstr(classes);

    fig = figure('Visible','off');
    try
        ch = confusionchart(cm, labels);   % <<< 關鍵：不要再傳 'Order'
        try ch.Normalization = "none"; end
        try ch.CellLabelFormat = "%d"; end
        try ch.Title = sprintf("%s | Acc=%.2f%%", name, acc*100); end

        % 匯出圖檔（舊版可能沒有 exportgraphics）
        outPng = fullfile(confDir, name + "_confusion.png");
        if exist("exportgraphics","file")
            exportgraphics(fig, outPng);
        else
            saveas(fig, outPng);
        end
    catch ME
        warning(ME.identifier, "confusionchart failed: %s", ME.message);
    end
    close(fig);

end

end
