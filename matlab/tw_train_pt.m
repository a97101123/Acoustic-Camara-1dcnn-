function tw_train_pt(cfg, task, val_knives)
% 輸出:
%  cnn2/train_ready2/<cond>_tr.mat
%   Xtrain, Ttrain, Xval, Tval

if nargin < 2, task = "stage_cls"; end
if nargin < 3, val_knives = [4 8 12 16 20]; end

conds = read_list_strings(cfg.files.micListXlsx);
outDir = cfg.paths.trainReady;
if ~isfolder(outDir), mkdir(outDir); end

% wear map（如果需要）
if task == "wear_cls"
    W = tw_build_wear_map(cfg);
end

for ci = 1:numel(conds)
    cond = conds(ci);
    if cond=="" || ismissing(cond), continue; end

    Xtrain = {}; Xval = {};
    Ttrain = categorical([]); Tval = categorical([]);

    for stage = 1:cfg.exp.numStages
        for knife = 1:cfg.exp.numKnives

            matFile = fullfile(cfg.paths.cnnRoot, cond + "_knife" + string(knife) + "_s" + string(stage) + ".mat");
            if ~isfile(matFile), continue; end

            S = load(matFile); % data, meta
            data = S.data;

            % label per sample
            if task == "stage_cls"
                y = repmat(categorical(stage), numel(data), 1);
            else % wear_cls
                h = double(S.meta.holeNum(:));
                y = W.class(h);
            end

            % split train/val by knife id
            if ismember(knife, val_knives)
                Xval   = [Xval;   data];
                Tval   = [Tval;   y];
            else
                Xtrain = [Xtrain; data];
                Ttrain = [Ttrain; y];
            end
        end
    end
    Ttrain = categorical(string(Ttrain), ["New","Light","Heavy"]);
    Tval   = categorical(string(Tval),   ["New","Light","Heavy"]);
    save(fullfile(outDir, cond + "_tr.mat"), "Xtrain","Ttrain","Xval","Tval","task","-v7.3");
    fprintf("Saved: %s\n", fullfile(outDir, cond + "_tr.mat"));
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
