cfg = tw_cfg("03m0d", struct('segMode',"holes",'holesPerWav',5,'micMode',"mean"));

netPath = fullfile(cfg.paths.trainReady, "models", "03m0d_noAC_tr_net.mat");

% 只選 wav
[files, path] = uigetfile({'*.wav','WAV files (*.wav)'}, '選擇要預測的音檔', 'MultiSelect','on');
if isequal(files,0)
    disp('使用者取消選擇');
    return;
end
if ischar(files), files = {files}; end

for i = 1:numel(files)
    wavPath = fullfile(path, files{i});
    out = tw_predict_wav(cfg, netPath, wavPath);
    fprintf('檔名: %s  ->  預測: %s\n', files{i}, string(out.final));
end


