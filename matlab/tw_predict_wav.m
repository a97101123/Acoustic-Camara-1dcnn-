function out = tw_predict_wav(cfg, netMatPath, wavPath)
% out.final        : 最終類別（投票）
% out.perChunkPred : 每個 chunk 的預測
% out.perChunkProb : 每個 chunk 的 softmax 機率（可用來做平均）

M = load(netMatPath);
net = M.net;

% 1) wav -> X (cell of [numFeatures x T])
X = tw_wav_to_X(cfg, wavPath);

% 2) predict
perChunkPred = classify(net, X, MiniBatchSize=32, SequencePaddingDirection="left");

% 若你想要機率
perChunkProb = predict(net, X, MiniBatchSize=32, SequencePaddingDirection="left");

% 3) aggregation (投票)
finalPred = mode(perChunkPred);

out = struct();
out.final = finalPred;
out.perChunkPred = perChunkPred;
out.perChunkProb = perChunkProb;
end


function X = tw_wav_to_X(cfg, wavPath)
[x, fs] = audioread(wavPath);

% fs 不一致就重採樣（很重要）
if fs ~= cfg.fs
    x = resample(x, cfg.fs, fs);
    fs = cfg.fs;
end

% mic array mode
if cfg.mic.mode == "mean"
    if size(x,2) > 1, x = mean(x,2); end
    Xch = x; % N×1
else
    Xch = x; % N×C
end

win = cfg.windowFcn(cfg.windowSize);
Pcat = [];
for ch = 1:size(Xch,2)
    [S,~,~] = stft(Xch(:,ch), fs, ...
        'Window', win, ...
        'OverlapLength', cfg.overlap, ...
        'FFTLength', cfg.nfft, ...
        'FrequencyRange','onesided');
    P = abs(S).^2;
    P = P(cfg.micFreqIdxAll, :);   % drop DC
    Pcat = [Pcat; P]; %#ok<AGROW>
end

% clip 爆點（跟訓練一致）
for r = 1:size(Pcat,1)
    mu = mean(Pcat(r,:), 'omitnan');
    Pcat(r, Pcat(r,:)>mu) = mu;
end

chunk_cols = cfg.chunk_cols;
X = {};

if cfg.exp.segMode == "holes"
    H = cfg.exp.holesPerWav;
    dura = floor(size(Pcat,2)/H);

    for u = 1:H
        cols = (u-1)*dura + (1:dura);
        cols = cols(cols<=size(Pcat,2));
        seg = Pcat(:, cols);

        cutL = floor(cfg.exp.edgeFrac * size(seg,2));
        if size(seg,2) > 2*cutL
            seg = seg(:, 1+cutL:end-cutL);
        end

        nChunks = floor(size(seg,2)/chunk_cols);
        for m = 1:nChunks
            one = seg(:, (m-1)*chunk_cols + (1:chunk_cols));
            X{end+1,1} = one; %#ok<AGROW>
        end
    end
else
    seg = Pcat;
    cutL = floor(cfg.exp.edgeFrac * size(seg,2));
    if size(seg,2) > 2*cutL
        seg = seg(:, 1+cutL:end-cutL);
    end

    nChunks = floor(size(seg,2)/chunk_cols);
    for m = 1:nChunks
        one = seg(:, (m-1)*chunk_cols + (1:chunk_cols));
        X{end+1,1} = one; %#ok<AGROW>
    end
end

if isempty(X)
    error("No chunks produced from wav: %s (可能太短或 chunkSec 太長)", wavPath);
end
end
