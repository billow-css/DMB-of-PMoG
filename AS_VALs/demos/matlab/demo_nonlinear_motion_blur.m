%% demo_nonlinear_motion_blur
% 空变非线性运动模糊实验（AS_VALs 600x600 → 512x512 取景窗）
%
% 思想：
%   - 起点位姿 g0：随机平移，旋转 θ0 = 0
%   - 终点位姿 g1：随机平移 + 随机旋转 θ
%   - 对曝光时间 t∈[0,1] 做刚体位姿插值 g(t)
%   - 每个输出像素沿自身轨迹双线性采样，高斯加权积分
%
% 输出四张 figure，并保存到 out/：
%   1) 原图 (600x600)
%   2) 结果图 (512 模糊)
%   3) 轨迹图 (位移场 + 抽样轨迹)
%   4) 起点/终点取景窗

clear; close all; clc;
rng(42);  % 固定种子，便于复现；改 [] 则每次随机

%% -------------------- 路径与输入 --------------------
thisDir = fileparts(mfilename('fullpath'));
imgDir  = fullfile(thisDir, '..', '..', 'outputs', 'text_images');
outDir  = fullfile(thisDir, '..', '..', 'outputs', 'demo_matlab');
if ~exist(outDir, 'dir'); mkdir(outDir); end

% 选一张 AS_VALs 图；也可改成具体文件名
imgName = '00010.png';  % manifest 里是 "new book"，文字较清晰
imgPath = fullfile(imgDir, imgName);
if ~isfile(imgPath)
    error('找不到图像: %s\n请确认 AS_VALs/outputs/text_images 下已有 PNG。', imgPath);
end

I = im2double(imread(imgPath));
if size(I, 3) == 1
    I = repmat(I, [1 1 3]);
end
[H, W, C] = size(I);
assert(H == 600 && W == 600, '期望 AS_VALs 为 600x600，实际为 %dx%d', H, W);

%% -------------------- 参数 --------------------
viewSize = 512;          % 取景窗边长
nSamples = 16;           % 轨迹时间采样数（越大越平滑、越慢）
pathPower = 2;           % 进度 λ=t^pathPower；2=匀加速(v∝t)，1=匀速
maxTrans = 28;           % 终点相对起点的最大平移（像素）
maxRotDeg = 10;          % 终点最大旋转角（度）
maxPoseTries = 500;      % 越界则重采样的最大次数
trajGrid = 16;           % 轨迹图上每隔多少像素画一条示意轨迹
quiverStep = 24;         % 位移场箭头稀疏步长

half = (viewSize - 1) / 2;   % 局部坐标相对窗中心：[-half, half]
% 取景窗四角（局部坐标），用于越界检查
cornerU = [-half,  half, half, -half];
cornerV = [-half, -half, half,  half];

%% -------------------- 随机起点 / 终点位姿（越界则重采） --------------------
% 全程（起点、终点、以及插值中间位姿）取景窗四角都必须落在 [1,W]×[1,H] 内
theta0 = 0;
ok = false;
nReject = 0;
for tryId = 1:maxPoseTries
    % 起点：θ=0，先保证轴对齐窗可放入图内
    cx0 = half + 1 + rand() * (W - viewSize);
    cy0 = half + 1 + rand() * (H - viewSize);

    % 终点：起点附近平移 + 旋转
    cx1 = cx0 + (2 * rand() - 1) * maxTrans;
    cy1 = cy0 + (2 * rand() - 1) * maxTrans;
    theta1 = deg2rad((2 * rand() - 1) * maxRotDeg);

    if pose_path_in_bounds(cornerU, cornerV, cx0, cy0, theta0, ...
            cx1, cy1, theta1, W, H, nSamples)
        ok = true;
        break;
    end
    nReject = nReject + 1;
end
if ~ok
    error(['连续 %d 次随机位姿均越界。请减小 maxTrans / maxRotDeg，' ...
           '或增大 maxPoseTries。'], maxPoseTries);
end

fprintf('图像: %s\n', imgPath);
fprintf('越界重采次数: %d\n', nReject);
fprintf('起点: cx=%.2f cy=%.2f theta=%.2f deg\n', cx0, cy0, rad2deg(theta0));
fprintf('终点: cx=%.2f cy=%.2f theta=%.2f deg\n', cx1, cy1, rad2deg(theta1));

%% -------------------- 局部网格（输出 512x512） --------------------
[uGrid, vGrid] = meshgrid(linspace(-half, half, viewSize), ...
                          linspace(-half, half, viewSize));

%% -------------------- 刚体插值（匀加速进度）+ 等权曝光 --------------------
tList = linspace(0, 1, nSamples);
lamList = tList .^ pathPower;   % λ(t)=t^p；p=2 → 静止起步匀加速
wList = ones(1, nSamples) / nSamples;

acc = zeros(viewSize, viewSize, C);
[x0map, y0map] = pose_map(uGrid, vGrid, cx0, cy0, theta0);
[x1map, y1map] = pose_map(uGrid, vGrid, cx1, cy1, theta1);

for k = 1:nSamples
    lam = lamList(k);
    cx = (1 - lam) * cx0 + lam * cx1;
    cy = (1 - lam) * cy0 + lam * cy1;
    th = (1 - lam) * theta0 + lam * theta1;
    [xs, ys] = pose_map(uGrid, vGrid, cx, cy, th);

    frame = zeros(viewSize, viewSize, C);
    for ch = 1:C
        % 位姿已保证不越界；extrapolate 值仅作兜底
        frame(:, :, ch) = interp2(I(:, :, ch), xs, ys, 'linear', 1);
    end
    acc = acc + wList(k) * frame;
end
Iblur = acc;

% 仅起点清晰裁切（对照）
Isharp = zeros(viewSize, viewSize, C);
for ch = 1:C
    Isharp(:, :, ch) = interp2(I(:, :, ch), x0map, y0map, 'linear', 1);
end

% 运动场：终点相对起点的位移（输出像素坐标系下）
dx = x1map - x0map;
dy = y1map - y0map;
mag = hypot(dx, dy);

%% -------------------- Figure 1: 原图 --------------------
fig1 = figure('Name', '1-原图', 'Color', 'w', 'Position', [80 80 640 680]);
imshow(I);
title(sprintf('原图 AS\\_VALs %s (600\\times600)', imgName), 'Interpreter', 'tex');
export_fig(fig1, fullfile(outDir, '01_original.png'));

%% -------------------- Figure 2: 结果图 --------------------
fig2 = figure('Name', '2-结果图', 'Color', 'w', 'Position', [100 100 980 520]);
subplot(1, 2, 1);
imshow(Isharp);
title('起点清晰取景 (512\times512)');
subplot(1, 2, 2);
imshow(Iblur);
title(sprintf('非线性运动模糊结果 (N=%d, p=%.2f)', nSamples, pathPower));
export_fig(fig2, fullfile(outDir, '02_result.png'));
imwrite(Iblur, fullfile(outDir, '02_result_only.png'));

%% -------------------- Figure 3: 轨迹图 --------------------
fig3 = figure('Name', '3-轨迹图', 'Color', 'w', 'Position', [120 60 1100 520]);

% 左：位移幅度热力图
subplot(1, 2, 1);
imagesc(mag); axis image; colormap(gca, parula); colorbar;
title(sprintf('逐像素轨迹长度 |p_1-p_0|  [%.2f, %.2f] px', min(mag(:)), max(mag(:))));
xlabel('u'); ylabel('v');

% 右：在原图上画抽样轨迹（刚体插值折线）
subplot(1, 2, 2);
imshow(I); hold on;
uu = uGrid(1:trajGrid:end, 1:trajGrid:end);
vv = vGrid(1:trajGrid:end, 1:trajGrid:end);
nShow = numel(uu);
tFine = linspace(0, 1, 12);
for i = 1:nShow
    ui = uu(i); vi = vv(i);
    xs = zeros(size(tFine));
    ys = zeros(size(tFine));
    for k = 1:numel(tFine)
        t = tFine(k);
        cx = (1 - t) * cx0 + t * cx1;
        cy = (1 - t) * cy0 + t * cy1;
        th = (1 - t) * theta0 + t * theta1;
        [xs(k), ys(k)] = pose_map(ui, vi, cx, cy, th);
    end
    plot(xs, ys, '-', 'Color', [0.1 0.45 0.95 0.85], 'LineWidth', 1.0);
    plot(xs(1), ys(1), 'g.', 'MarkerSize', 8);
    plot(xs(end), ys(end), 'r.', 'MarkerSize', 8);
end
% 稀疏 quiver（起点处）
[xq, yq] = pose_map(uGrid(1:quiverStep:end, 1:quiverStep:end), ...
                    vGrid(1:quiverStep:end, 1:quiverStep:end), cx0, cy0, theta0);
dxq = x1map(1:quiverStep:end, 1:quiverStep:end) - xq;
dyq = y1map(1:quiverStep:end, 1:quiverStep:end) - yq;
quiver(xq, yq, dxq, dyq, 0, 'y', 'LineWidth', 0.8);
title('原图上的抽样轨迹 (绿=起点, 红=终点, 黄=位移箭头)');
hold off;
export_fig(fig3, fullfile(outDir, '03_trajectory.png'));

%% -------------------- Figure 4: 起点 / 终点取景窗 --------------------
fig4 = figure('Name', '4-起点终点', 'Color', 'w', 'Position', [140 80 700 720]);
imshow(I); hold on;
corners = [-half -half; half -half; half half; -half half; -half -half];
poly0 = zeros(5, 2);
poly1 = zeros(5, 2);
for i = 1:5
    [poly0(i, 1), poly0(i, 2)] = pose_map(corners(i, 1), corners(i, 2), cx0, cy0, theta0);
    [poly1(i, 1), poly1(i, 2)] = pose_map(corners(i, 1), corners(i, 2), cx1, cy1, theta1);
end
plot(poly0(:, 1), poly0(:, 2), 'g-', 'LineWidth', 2.2);
plot(poly1(:, 1), poly1(:, 2), 'r-', 'LineWidth', 2.2);
plot(cx0, cy0, 'g+', 'MarkerSize', 14, 'LineWidth', 1.5);
plot(cx1, cy1, 'rx', 'MarkerSize', 12, 'LineWidth', 1.5);
% 中心运动轨迹
tFine = linspace(0, 1, 40);
cxs = (1 - tFine) * cx0 + tFine * cx1;
cys = (1 - tFine) * cy0 + tFine * cy1;
plot(cxs, cys, 'c--', 'LineWidth', 1.4);
legend({'起点取景窗 \theta=0', sprintf('终点取景窗 \\theta=%.1f°', rad2deg(theta1)), ...
        '起点中心', '终点中心', '中心轨迹'}, ...
       'Location', 'southoutside', 'NumColumns', 2);
title('起点 / 终点取景窗叠加在原图上');
hold off;
export_fig(fig4, fullfile(outDir, '04_start_end.png'));

%% -------------------- 参数摘要 --------------------
fid = fopen(fullfile(outDir, 'run_info.txt'), 'w');
fprintf(fid, 'image=%s\n', imgPath);
fprintf(fid, 'viewSize=%d\n', viewSize);
fprintf(fid, 'nSamples=%d\n', nSamples);
fprintf(fid, 'pathPower=%.4f\n', pathPower);
fprintf(fid, 'reject_count=%d\n', nReject);
fprintf(fid, 'start: cx=%.4f cy=%.4f theta_deg=%.4f\n', cx0, cy0, rad2deg(theta0));
fprintf(fid, 'end:   cx=%.4f cy=%.4f theta_deg=%.4f\n', cx1, cy1, rad2deg(theta1));
fprintf(fid, 'mag_px: min=%.4f max=%.4f mean=%.4f\n', min(mag(:)), max(mag(:)), mean(mag(:)));
fclose(fid);

fprintf('\n已保存到: %s\n', outDir);
fprintf('  01_original.png\n  02_result.png / 02_result_only.png\n');
fprintf('  03_trajectory.png\n  04_start_end.png\n  run_info.txt\n');

%% ========================================================================
function [X, Y] = pose_map(U, V, cx, cy, theta)
% 局部坐标 (U,V)（相对取景窗中心）→ 原图坐标 (X,Y)
ct = cos(theta);
st = sin(theta);
X = ct .* U - st .* V + cx;
Y = st .* U + ct .* V + cy;
end

function tf = pose_in_bounds(U, V, cx, cy, theta, W, H)
% 取景窗采样点是否全部落在图像内（含边界）
[X, Y] = pose_map(U, V, cx, cy, theta);
tol = 1e-6;
tf = all(X(:) >= 1 - tol & X(:) <= W + tol & ...
         Y(:) >= 1 - tol & Y(:) <= H + tol);
end

function tf = pose_path_in_bounds(U, V, cx0, cy0, theta0, cx1, cy1, theta1, W, H, nCheck)
% 检查起点、终点及刚体插值中间位姿是否都不越界
if nargin < 11 || isempty(nCheck)
    nCheck = 16;
end
tList = linspace(0, 1, nCheck);
tf = true;
for k = 1:numel(tList)
    t = tList(k);
    cx = (1 - t) * cx0 + t * cx1;
    cy = (1 - t) * cy0 + t * cy1;
    th = (1 - t) * theta0 + t * theta1;
    if ~pose_in_bounds(U, V, cx, cy, th, W, H)
        tf = false;
        return;
    end
end
end

function export_fig(figHandle, filePath)
% 优先 exportgraphics；旧版 MATLAB 回退 print
if exist('exportgraphics', 'file') == 2
    exportgraphics(figHandle, filePath, 'Resolution', 150);
else
    set(figHandle, 'PaperPositionMode', 'auto');
    print(figHandle, filePath, '-dpng', '-r150');
end
fprintf('saved: %s\n', filePath);
end
