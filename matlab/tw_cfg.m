function cfg = tw_cfg(datasetTag, common)
%TW_CFG 你自己的設定檔（仿 toolwear_cfg.m，但不限制 datasetTag）
% datasetTag: "5DRILL" 或 "LATHE"
% common: 可選，struct 覆蓋預設參數

datasetTag = string(datasetTag);

%% ===== defaults =====
opts = struct();

% --- sampling / STFT ---
opts.fs = 48000;
opts.nfft = 2048;
opts.windowSize = 1024;
opts.overlapRatio = 0.75;

% --- wave cut ---
opts.cutDurationSec = 67;

% --- chunking (CNN time steps) ---
opts.chunkSec = 5;

% --- experiment layout (你一定要依資料改) ---
opts.numKnives   = 21;     % 你的刀具/樣本組數（請改成你實際的）
opts.numStages   = 3;      % 5DRILL 多半 3；車刀你有幾段就填幾
opts.holesPerWav = 5;      % 5DRILL = 5；車刀通常 = 1（continuous）

% --- segmentation mode ---
opts.segMode = "holes";    % "holes" or "continuous"
opts.edgeFrac = 0.2;       % 每段前後切掉比例（你原本是 0.2）

% --- mic array mode ---
opts.micMode = "mean";     % "mean"=多通道平均成單通道；"stack"=通道特徵疊起來

% --- wear label (如果你要 wear_cls) ---
opts.wearExcel = "flank wear.xlsx";
opts.wearRow   = 10;
opts.wearScale = 1.0;
opts.wearSevMm = [0.10 0.20]; % New/Light/Heavy 門檻

%% ===== merge common overrides =====
if nargin >= 2 && isstruct(common)
    fns = fieldnames(common);
    for i = 1:numel(fns)
        opts.(fns{i}) = common.(fns{i});
    end
end

%% ===== build cfg =====
cfg = struct();


cfg.datasetTag = datasetTag;

cfg.fs = opts.fs;
cfg.windowSize = opts.windowSize;
cfg.nfft = opts.nfft;
cfg.overlap = round(cfg.windowSize * opts.overlapRatio);
cfg.hop = cfg.windowSize - cfg.overlap;
cfg.windowFcn = @(L) hann(L,"periodic");
cfg.df = cfg.fs / cfg.nfft;
cfg.cutDurationSec = opts.cutDurationSec;
% one-sided bins, drop DC
cfg.micFreqIdxAll = 2:(cfg.nfft/2 + 1);
cfg.micAllBins = numel(cfg.micFreqIdxAll);

% chunk cols in STFT frames
cfg.chunkSec = opts.chunkSec;
cfg.chunk_cols = max(1, round(cfg.chunkSec * cfg.fs / cfg.hop));

% experiment
cfg.exp = struct();
cfg.exp.numKnives   = opts.numKnives;
cfg.exp.numStages   = opts.numStages;
cfg.exp.holesPerWav = opts.holesPerWav;
cfg.exp.segMode     = string(opts.segMode);
cfg.exp.edgeFrac    = opts.edgeFrac;
cfg.exp.numHoles = cfg.exp.numStages * cfg.exp.holesPerWav;
cfg.exp.hole_num = (1:cfg.exp.numHoles)';
% mic
cfg.mic = struct();
cfg.mic.mode = string(opts.micMode);

% wear
cfg.wear = struct();
cfg.wear.excel = string(opts.wearExcel);
cfg.wear.row = opts.wearRow;
cfg.wear.scale = opts.wearScale;
cfg.wear.sevMm = opts.wearSevMm;
cfg.wear.measuredHoles = [1 5 6 10 11 15]';   % 你這份 flank wear.xlsx 一列 6 個值對應的孔位
% paths
cfg.paths = struct();
cfg.paths.acousticRoot = fullfile('.', 'acoustic');
cfg.paths.wavecutRoot  = fullfile('.', 'wavecut');
cfg.paths.cnnRoot      = fullfile('.', 'cnn2');
cfg.paths.trainReady   = fullfile(cfg.paths.cnnRoot, 'train_ready2');

% files
cfg.files = struct();
cfg.files.fileNameXlsx = fullfile('.', "file_name_" + datasetTag + ".xlsx");
cfg.files.micCutXlsx   = fullfile(cfg.paths.wavecutRoot, "mic_wave_cut_" + datasetTag + ".xlsx");
cfg.files.micListXlsx  = fullfile('.', "mic_list_" + datasetTag + ".xlsx");

% ensure folders
if ~isfolder(cfg.paths.wavecutRoot), mkdir(cfg.paths.wavecutRoot); end
if ~isfolder(cfg.paths.cnnRoot), mkdir(cfg.paths.cnnRoot); end
if ~isfolder(cfg.paths.trainReady), mkdir(cfg.paths.trainReady); end

end
