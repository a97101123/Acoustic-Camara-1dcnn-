function W = tw_build_wear_map(cfg)
%TW_BUILD_WEAR_MAP 由 flank wear.xlsx 建立 1..numHoles 的 wear(mm) + wear class
% 支援：
%  - 若 excel 該列剛好有 numHoles 個值：直接用
%  - 若 excel 該列只有 numel(measuredHoles) 個值：用 interp1 補齊所有孔位

wearFile = cfg.wear.excel;
row      = cfg.wear.row;
scale    = cfg.wear.scale;

numHoles = cfg.exp.numHoles;
hole_num = cfg.exp.hole_num(:);

measuredHoles = cfg.wear.measuredHoles(:);

M = readmatrix(wearFile);
if row > size(M,1)
    error("wearRow=%d 超過 wear.xlsx 的列數=%d", row, size(M,1));
end

wRow = M(row, :);
wRow = wRow(~isnan(wRow));
wRow = double(wRow) * scale;

% --- 兩種合法情況：全孔位 or 量測孔位 ---
if numel(wRow) == numHoles
    wear_mm = reshape(wRow, [], 1);
elseif numel(wRow) == numel(measuredHoles)
    wear_mm = interp1(double(measuredHoles), wRow(:), double(hole_num), 'linear', 'extrap');
else
    error("wear 資料長度=%d 不符合：需要 numHoles=%d 或 measuredHoles=%d", ...
        numel(wRow), numHoles, numel(measuredHoles));
end

% --- class ---
t1 = cfg.wear.sevMm(1);
t2 = cfg.wear.sevMm(2);

cls = strings(numHoles,1);
for i = 1:numHoles
    if wear_mm(i) < t1
        cls(i) = "New";
    elseif wear_mm(i) < t2
        cls(i) = "Light";
    else
        cls(i) = "Heavy";
    end
end

W = struct();
W.mm = wear_mm(:);
W.class = categorical(cls, ["New","Light","Heavy"]);
end
