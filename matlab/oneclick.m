clear; clc;close all

% ===== 5DRILL =====
datasetTags = ["03m0d", "06m0d", "06m15d"];
disp('=== START TRAIN ===');
for di = 1:numel(datasetTags)
    tag = datasetTags(di);
    fprintf('\n==============================\n');
    fprintf('== DATASET: %s\n', tag);
    fprintf('==============================\n');

    cfg = tw_cfg(tag, struct('segMode',"holes",'holesPerWav',5,'micMode',"mean",'numKnives',21,'numStages',3));
    disp('=== wave cut ===');
    tw_wave_cut_mic(cfg);
    disp('=== DONE ===');
    disp('=== traincellgen ===');
    tw_mic_traincellgen_freq(cfg);
    disp('=== DONE ===');
    disp('=== train_pt_generate_auto ===');
    tw_train_pt(cfg, "stage_cls");   % 先跑 stage_cls 再 waer_cls
    disp('=== DONE ===');
    fprintf('==============================\n');
    fprintf('== DONE DATASET: %s\n', tag);
    fprintf('==============================\n');
end
disp('=== train_1dcnn_auto ===');
tw_train_1dcnn(cfg);
disp('=== DONE ===');
disp('=== ALL DONE ===');