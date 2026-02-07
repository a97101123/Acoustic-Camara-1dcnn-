function foldInfo = tw_train_pt_kfold(cfg, task, K, varargin)
%TW_TRAIN_PT_KFOLD Build K folds by KNIFE (group K-fold) to reduce leakage/overfit.
%
% Usage:
%   foldInfo = tw_train_pt_kfold(cfg, "stage_cls", 5);
%   foldInfo = tw_train_pt_kfold(cfg, "wear_cls",  5, "seed", 42);
%
% Output folders:
%   <cfg.paths.trainReady>/kfold/fold01/<cond>_tr.mat
%   <cfg.paths.trainReady>/kfold/fold02/<cond>_tr.mat
%   ...
%
% Each *_tr.mat contains: Xtrain,Ttrain,Xval,Tval,task,val_knives

arguments
    cfg (1,1) struct
    task (1,1) string {mustBeMember(task,["stage_cls","wear_cls"])}
    K (1,1) double {mustBeInteger, mustBePositive} = 5
end
arguments (Repeating)
    varargin
end

% ---- options ----
p = inputParser;
addParameter(p, "seed", 1);
parse(p, varargin{:});
seed = p.Results.seed;

% ---- folds by knife ----
numKnives = cfg.exp.numKnives;
folds = makeKnifeFolds(numKnives, K, seed);

% ---- conditions ----
conds = read_list_strings(cfg.files.micListXlsx);

% ---- wear map (if needed) ----
if task == "wear_cls"
    numHoles = cfg.exp.numStages * cfg.exp.holesPerWav;
    W = tw_build_wear_map(cfg, numHoles); % local helper below
end

baseOut = fullfile(cfg.paths.trainReady, "kfold");
if ~isfolder(baseOut), mkdir(baseOut); end

foldInfo = struct();
foldInfo.K = K;
foldInfo.seed = seed;
foldInfo.folds = folds;

for f = 1:K
    val_knives = folds{f};
    foldDir = fullfile(baseOut, sprintf("fold%02d", f));
    if ~isfolder(foldDir), mkdir(foldDir); end

    fprintf("\n==== K-FOLD %d/%d | val knives = [%s] ====\n", ...
        f, K, strjoin(string(val_knives), ","));

    for ci = 1:numel(conds)
        cond = conds(ci);
        if cond=="" || ismissing(cond), continue; end

        Xtrain = {}; Xval = {};
        Ttrain = categorical([]); Tval = categorical([]);

        for stage = 1:cfg.exp.numStages
            for knife = 1:cfg.exp.numKnives
                matFile = fullfile(cfg.paths.cnnRoot, cond + "_knife" + string(knife) + "_s" + string(stage) + ".mat");
                if ~isfile(matFile), continue; end

                S = load(matFile); % data, meta(optional)
                data = S.data;     % cell; each is (nFeat x T)

                % ----- labels per sample -----
                if task == "stage_cls"
                    y = repmat(categorical(stage), numel(data), 1);
                else
                    if ~isfield(S,'meta') || ~isfield(S.meta,'holeNum')
                        error("wear_cls requires meta.holeNum in %s", matFile);
                    end
                    h = double(S.meta.holeNum(:));
                    if numel(h) < numel(data)
                        error("meta.holeNum shorter than data: %s", matFile);
                    end
                    y = W.class(h(1:numel(data)));
                end

                % ----- split by knife -----
                if ismember(knife, val_knives)
                    Xval   = [Xval;   data];
                    Tval   = [Tval;   y];
                else
                    Xtrain = [Xtrain; data];
                    Ttrain = [Ttrain; y];
                end
            end
        end

        outFile = fullfile(foldDir, cond + "_tr.mat");
        save(outFile, "Xtrain","Ttrain","Xval","Tval","task","val_knives","-v7.3");
        fprintf("Saved: %s\n", outFile);
    end
end

end

%% ================= helpers =================

function folds = makeKnifeFolds(numKnives, K, seed)
% No Stats Toolbox needed. Balanced-ish round-robin split after shuffle.
rng(seed, "twister");
ids = 1:numKnives;
ids = ids(randperm(numKnives));

folds = cell(K,1);
for k = 1:K
    folds{k} = ids(k:K:end);
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

function W = tw_build_wear_map_simple(cfg, numHoles)
% Minimal wear map (你之後可換成你原本 toolwear_buildToolWear 的插值版)
M = readmatrix(cfg.wear.excel);
w = double(M(cfg.wear.row, :));
w = w(~isnan(w)) * cfg.wear.scale;

if numel(w) < numHoles
    error("wear 資料不足：需要至少 %d 個值", numHoles);
end
wear_mm = w(1:numHoles);

t1 = cfg.wear.sevMm(1); t2 = cfg.wear.sevMm(2);
cls = strings(1,numHoles);
for i = 1:numHoles
    if wear_mm(i) < t1
        cls(i) = "light";
    elseif wear_mm(i) < t2
        cls(i) = "medium";
    else
        cls(i) = "heavy";
    end
end
W = struct();
W.mm = wear_mm(:);
W.class = categorical(cls, ["light","medium","heavy"]);
end
