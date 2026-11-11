function tw_mic_traincellgen_freq(cfg)
%TW_MIC_TRAINCELLGEN_FREQ 從 *_cut.wav 產生 cnn2/*.mat
%輸出:
%  cnn2/<cond>_knife<k>_s<s>.mat
%   - data{m}: (nFeat x chunk_cols)
%   - meta.holeNum(m): 每個 chunk 對應的(全域)孔號/段號（wear 任務會用到）

fnm   = readmatrix(cfg.files.fileNameXlsx, "OutputType","string");
conds = read_list_strings(cfg.files.micListXlsx);
[fc, fr] = size(fnm);

win = cfg.windowFcn(cfg.windowSize);
chunk_cols = cfg.chunk_cols;

for c = 1:numel(conds)
    cond = conds(c);
    if cond=="" || ismissing(cond), continue; end

    for stage = 1:fr
        for knife = 1:fc
            base = fnm(knife,stage);
            if base=="" || ismissing(base), continue; end

            fnx = fullfile(cfg.paths.acousticRoot, cond, base + "_cut.wav");
            if ~isfile(fnx)
                warning("Missing cut wav: %s", fnx);
                continue;
            end

            [x, fs] = audioread(fnx);
            if fs ~= cfg.fs
                warning("fs mismatch: %d vs %d (%s)", fs, cfg.fs, fnx);
            end

            % ===== mic array handling =====
            % x: N x C
            if cfg.mic.mode == "mean"
                if size(x,2) > 1, x = mean(x,2); end
                Xch = x; % N x 1
            else % "stack"
                Xch = x; % N x C
            end

            % ===== STFT => P (features x frames) =====
            Pcat = [];
            numCh = size(Xch,2);
            for ch = 1:numCh
                [S,~,~] = stft(Xch(:,ch), fs, ...
                    'Window', win, ...
                    'OverlapLength', cfg.overlap, ...
                    'FFTLength', cfg.nfft, ...
                    'FrequencyRange','onesided');
                P = abs(S).^2;                 % (nfft/2+1) x nFrames
                P = P(cfg.micFreqIdxAll, :);   % drop DC => nBins x nFrames
                Pcat = [Pcat; P]; %#ok<AGROW>  % stack along feature dimension
            end

            % ===== clipping (避免少數爆點) =====
            for r = 1:size(Pcat,1)
                mu = mean(Pcat(r,:), 'omitnan');
                Pcat(r, Pcat(r,:)>mu) = mu;
            end

            % ===== segmentation =====
            data = {};
            holeNums = [];

            if cfg.exp.segMode == "holes"
                H = cfg.exp.holesPerWav;                 % 5DRILL: 5
                dura = floor(size(Pcat,2)/H);
                for u = 1:H
                    cols = (u-1)*dura + (1:dura);
                    cols = cols(cols<=size(Pcat,2));
                    seg = Pcat(:, cols);

                    % cut edges
                    cutL = floor(cfg.exp.edgeFrac * size(seg,2));
                    if size(seg,2) > 2*cutL
                        seg = seg(:, 1+cutL:end-cutL);
                    end

                    % chunk within this hole (避免跨孔)
                    nChunks = floor(size(seg,2)/chunk_cols);
                    for m = 1:nChunks
                        one = seg(:, (m-1)*chunk_cols + (1:chunk_cols));
                        data{end+1,1} = one; %#ok<AGROW>

                        % 全域孔號：把 stage 也算進去（stage=1..fr）
                        globalHole = (stage-1)*H + u;
                        holeNums(end+1,1) = globalHole; %#ok<AGROW>
                    end
                end
            else
                % continuous: 不切孔，直接全段 chunk
                seg = Pcat;
                cutL = floor(cfg.exp.edgeFrac * size(seg,2));
                if size(seg,2) > 2*cutL
                    seg = seg(:, 1+cutL:end-cutL);
                end

                nChunks = floor(size(seg,2)/chunk_cols);
                for m = 1:nChunks
                    one = seg(:, (m-1)*chunk_cols + (1:chunk_cols));
                    data{end+1,1} = one; %#ok<AGROW>
                    holeNums(end+1,1) = stage; %#ok<AGROW> % 至少給個可對齊的 index
                end
            end

            if isempty(data)
                warning("No chunks produced: %s", fnx);
                continue;
            end

            meta = struct();
            meta.holeNum = holeNums;   % wear_cls / wear_reg 會用到
            meta.knife   = knife;
            meta.stage   = stage;

            outMat = fullfile(cfg.paths.cnnRoot, cond + "_knife" + string(knife) + "_s" + string(stage) + ".mat");
            save(outMat, 'data', 'meta', '-v7.3');
        end
    end
end

end

function lst = read_list_strings(xlsxPath)
C = readcell(xlsxPath); C = C(:);
out = strings(0,1);
for i = 1:numel(C)
    v = C{i};
    if isempty(v), continue; end
    if isnumeric(v) && isscalar(v) && isnan(v), continue; end
    s = strtrim(string(v));
    if strlength(s)==0, continue; end
    out(end+1,1) = s; %#ok<AGROW>
end
[~,ia] = unique(out,'stable');
lst = out(ia);
end
