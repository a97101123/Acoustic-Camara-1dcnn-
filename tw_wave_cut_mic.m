function tw_wave_cut_mic(cfg)
%TW_WAVE_CUT_MIC 依照 mic_wave_cut_*.xlsx 把原始 wav 切成 *_cut.wav

fnm = readmatrix(cfg.files.fileNameXlsx, "OutputType","string");
mic_cut = readmatrix(cfg.files.micCutXlsx);
conds = read_list_strings(cfg.files.micListXlsx);

[fc, fr] = size(fnm);

for c = 1:numel(conds)
    cond = conds(c);
    if cond=="" || ismissing(cond), continue; end

    inDir = fullfile(cfg.paths.acousticRoot, cond);
    if ~isfolder(inDir)
        warning("Missing folder: %s", inDir); 
        continue;
    end

    for i = 1:fc
        for j = 1:fr
            base = fnm(i,j);
            if base=="" || ismissing(base), continue; end

            tStart = double(mic_cut(i,j));
            if ~isfinite(tStart), continue; end

            inFile  = fullfile(inDir, base + ".wav");
            outFile = fullfile(inDir, base + "_cut.wav");
            if ~isfile(inFile)
                warning("Missing wav: %s", inFile);
                continue;
            end

            range = round([tStart*cfg.fs, (tStart + cfg.cutDurationSec)*cfg.fs]);
            [x, fs] = audioread(inFile, range);
            if fs ~= cfg.fs
                warning("fs mismatch: %d vs %d (%s)", fs, cfg.fs, inFile);
            end

            % 不在這一步強制 mono，留給下一步（traincellgen）做 mic array 處理
            audiowrite(outFile, x, fs);
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
