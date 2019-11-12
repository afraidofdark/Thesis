clear all;
close all;
clc;
format long;

fname = "c1";
writeToFile = 0;
genTruth = 0;

img = imread(fname+".jpg");
img = rgb2gray(img);
img = im2double(img);
oimg = img; % save for later

% Blur and hist eq.
img = imgaussfilt(img, 1.4);
img = adapthisteq(img);
%img = imsharpen(img,'Radius',5,'Amount',1);

img = img + 0.001; % shift the image epsilon for 0 checks

% Cartesian image
cimg = flipud(img);

[rs, cs] = size(cimg);
hs = floor(sqrt(rs^2+cs^2));

inc = 1;
d = (1:inc:180);
histArr = zeros(length(d), hs);
dfs = zeros(length(d), 1);

cmask = true(size(cimg));
parfor t = 1 : length(d)    
    % rotate image clear border
    Irot = imrotate(cimg, d(t));
    Mrot = ~imrotate(cmask, d(t));
    Irot(Mrot&~imclearborder(Mrot)) = nan;
    
    % normalize histogram
    b = sum(isnan(Irot) == 0);
    p = length((b > 0)); % projection length instead cos(deg2rad(d(t))
    h = sum(Irot, 'omitnan');
    b = sum(isnan(Irot) == 0);
    b(b == 0) = 1; % Prevent divide by zero.
    h = h ./ b;
    
    % convert to probability and find entrophy
    h = h / sum(h);
    
    % Fill histogram
    myhist = zeros(1, hs);
    myhist(1:length(h)) = h;
    myhist(length(h):end) = 0;
    histArr(t, :) = myhist;
    
    % Calc max freq amp
    da = abs(fft(lowpass(h, 0.001)));
    dfs(t) = max(da(2:end));
end

%% Estimate Weft and Warp alignments

figure;
subplot(2,1,1);
plot(dfs);
title('FFT');
degInterval = 15;
[val, deg, dep] = nonmaxsup(-dfs', round(degInterval/inc, 0));

xs = [0,0;0,0];
ys = [0,0;0,0];
slopes = [0,0];
yarnWidths = [0, 0];
integralLines = [{0}, {0}];

% DFT angle search
anglesDetected = 0;
for i = 1:length(dep)
    if anglesDetected
        break;
    end
    
    [mv, mi] = max(dep);
    dep(mi) = -inf; % find max depth of all
    
    h = histArr(deg(mi) * inc, :);
    h = h(h>0);
    h = lowpass(h(h>0), 0.001);
    integralLines(1) = {h};
    fd = abs(fft(h));
    [~, yarnWidths(1)] = max(fd(2:end));
    
    nextAngle = (deg(mi) * inc) + 90; % search Deg
    if nextAngle > 180 % Circle angle
        nextAngle = nextAngle - 180;
    end
    if nextAngle < 1
        nextAngle = 180 + nextAngle;
    end
    
    deepest = -inf;
    for j = 1:length(dep)
        if j == mi
            continue;
        end
        
        checkRange = 10;
        angle = deg(j) * inc;
        if (angle > nextAngle - checkRange) && (angle < nextAngle + checkRange) % angle whitin range
            anglesDetected = 1;
            if deepest < dep(j)
                deepest = dep(j);
                
                h = histArr(angle,:);
                h = lowpass(h(h>0), 0.001);
                integralLines(2) = {h};
                fd = abs(fft(h));
                [~, yarnWidths(2)] = max(fd(2:end));
                
                slopes = [(deg(mi) * inc), angle];
                radi = min(cs, rs) * 0.3;
                for k=1:2
                    xs(k,:) = [cs * 0.5, cs * 0.5 + cos(deg2rad(slopes(k))) * radi];
                    ys(k,:) = [rs * 0.5, rs * 0.5 + sin(deg2rad(slopes(k))) * radi];
                end
            end
        end
    end
end

if ~anglesDetected
    disp "angles not detected";
    return;
end

% Sort angles vertical
diffAngs = [0, 0];
for i = 1:2
    diffAngs(i) = abs(0 - slopes(i));
    if diffAngs(i) > abs(180 - slopes(i))
        diffAngs(i) = abs(180 - slopes(i));
    end
end

% Switch slopes if necessery.
if diffAngs(1) > diffAngs(2)
    tmp = slopes(1);
    slopes(1) = slopes(2);
    slopes(2) = tmp;
    
    tmp = integralLines{1};
    integralLines{1} = integralLines{2};
    integralLines{2} = tmp;
    
    tmp = yarnWidths(1);
    yarnWidths(1) = yarnWidths(2);
    yarnWidths(2) = tmp;
end

yarnWidths = yarnWidths + 1; % count dc
yarnWidths = [length(integralLines{1}), length(integralLines{2})] ./ yarnWidths; % Freq to period (t = (1/f)*lengthofsignal)

hold on
scatter(slopes, dfs(slopes)); % plot indices

% slopes are projection line angle + 90 degree to make it yarn angle.
% slopes = slopes + 90; % calculate yarn angles angles
% slopes = slopes + (slopes>180) * -180;

subplot(2,1,2);
imshow(cimg);

hold on
plot(xs',ys','r-*','linewidth',1.5);
set(gca,'ydir','normal');

figure
plot(integralLines{1,1});
figure
plot(integralLines{1,2});

%% Plot grid

estYarnLocs = cell(2,1);
gridLines = cell(2,1);
figure;
imshow(cimg);
for i = 1:2
    slope = slopes(i);
    curve = integralLines{i};
    [~,bottoms,~] = nonmaxsup(curve, round(yarnWidths(i)*0.9, 0));
    
    Irot = imrotate(cimg, slope);
    [rsr, rsc] = size(Irot);
    xs = repelem(bottoms,2);
    xs = reshape(xs, 2, []);
    xs(3,:) = nan;
    xs = reshape(xs, 1, []);
    ys = repmat([0,rsr], 1, length(bottoms));
    ys = reshape(ys, 2, []);
    ys(3,:) = nan;
    ys = reshape(ys, 1, []);
    
    lines = [xs; ys];
    
    % Rotate grid lines to image space
    rang = deg2rad(slope);
    [rc1, cc1] = size(cimg);
    ofy = rc1 - rsr;
    ofx = cc1 - rsc;
    orginDelta = [ofx; ofy] * 0.5;
    
    shiftv = [rsc; rsr] * 0.5; % rotate around image origin.
    R = [cos(rang) -sin(rang); sin(rang) cos(rang)];
    lines = R * (lines - shiftv) + shiftv + orginDelta;
    gridLines{i} = lines;
    
    c = 'green';
    if i == 2
        c = 'red';
    end
    
    hold on;
    line(lines(1,:),lines(2,:), 'color', c);
    set(gca,'ydir','normal');
end

%% Find intersections

lines1 = gridLines{1};
xs = lines1(1,:);
xs = xs(~isnan(xs));
ys = lines1(2,:);
ys = ys(~isnan(ys));
lines1 = [xs;ys];

lines2 = gridLines{2};
xs = lines2(1,:);
xs = xs(~isnan(xs));
ys = lines2(2,:);
ys = ys(~isnan(ys));
lines2 = [xs;ys];

figure;
imshow(flipud(oimg));
fig = gcf;
altFig = -1;

capImg = cimg;
subcenters = [];
imgs = {};

checkMap = ones(int64(length(lines2) / 2), int64(length(lines1) / 2)) * -1;
for i = 4:2:length(lines1)
    line1 = lines1(:, [i-1,i]);
    line2 = lines1(:, [i-3,i-2]);
    hfi = 0;
    hli = 0;
    for j = 4:2:length(lines2)
        line3 = lines2(:, [j-1,j]);
        line4 = lines2(:, [j-3,j-2]);
        
        xx = intersectLines(line1, line2, line3, line4);
        if (length(xx) ~= 4)
            continue;
        end
        
        breakit = false;
        for k = 1:4
            if ~withinBoundary([cs,rs], xx(k,:))
                breakit = true;
                break;
            end
        end
        
        if breakit
            continue;
        end
        
        % capture section.
        if hfi == 0
           hfi = j; 
        end
        hli = j;
        
        % stable sort, result upperleft corner, CW.
        xx = sortULCW(xx);

        line([xx(1,1),xx(2,1),xx(4,1),xx(3,1),xx(1,1)], [xx(1,2),xx(2,2),xx(4,2),xx(3,2),xx(1,2)], 'Color', 'Red');
        subcenters = [subcenters; mean(xx)];
        
        marg = 2;
        pixSize = 60 + marg * 2;
        mapped = zeros(pixSize);
        for kk=0:(1/pixSize):0.99
            for jj=0:(1/pixSize):0.99
                lup = int64(blinMap([jj,kk], xx));
                yk = int64(kk*pixSize+1);
                xj = int64(jj*pixSize+1);
                mapped(xj, yk) = capImg(lup(2), lup(1));
            end
        end
        
        imgs{end + 1} = flipud(mapped);
        checkMap(int64(j/2),int64(i/2)) = length(imgs);
        
        if genTruth
            title(genTruth);
            if genTruth == 1
                figure
                altFig = gcf;
            end
            set(0, 'CurrentFigure', altFig)
            imshow(flipud(mapped));
            set(0, 'CurrentFigure', fig)
            
            ginput(1);
            genTruth = genTruth+1;
            continue;
        end
    end
    
    if hfi > 0 && hli > 0
        line3 = lines2(:, [hfi-3,hfi-2]);
        line4 = lines2(:, [hli-1,hli]);        
        
        xx = intersectLines(line1, line2, line3, line4);
        xx = sortULCW(xx);
        sectImg = mapToRectangle(cimg, xx);
    end
    
    hfi = 0;
    hli = 0;
end

%% Mark clustering

features = [];
thruthVals = [];
fid = fopen('cropped\'+fname+'_table.txt');

for i=1:length(imgs)
    img = imgs{i};
    stats = appGlcm(img);
    features = [features; stats];
    truthVal = str2double(fgetl(fid));
    
    if writeToFile > 0
        sub = "downs\\";
        if truthVal
            sub = "ups\\";
        end
        
        imwrite((img(marg:pixSize-marg-1,marg:pixSize-marg-1)), "cropped\\"+sub+fname+"_"+string(writeToFile)+".jpg");
        writeToFile = writeToFile + 1;
    end
    
    thruthVals = [thruthVals, truthVal];
end

%xfeatures = features - mean(features); % zero mean
%xfeatures = (xfeatures - min(xfeatures)) ./ (max(xfeatures) - min(xfeatures)); % 0 - 1 scale

labelFeatures = features(:,end-19:end);
features = features(:,1:end-20);

xfeatures = StatisticalNormaliz(features,'standard');

% PCA
numberOfDimensions = 8;
coeff = pca(xfeatures);
reducedDimension = coeff(:,1:numberOfDimensions);
reducedData = xfeatures * reducedDimension;
xfeatures = reducedData;

[centers,U] = ffcmw(xfeatures,2);

maxU = max(U);
cluster = (U(1,:) == maxU);

% Learn cluster labels
[~, confIds] = sort(U(1,:), 'descend');

% Ask labels of high confided n sample
up = 0;
down = 0;
for i=1:4
    figure
    imshow(imgs{confIds(i)});
    r = inputdlg("0 - 1");
    close
    up = up + (r{1}=='1');
    down = down + (r{1}=='0');
end

totalError = sum(cluster ~= thruthVals);
if down > up
    cluster = ~cluster;
    totalError = sum(cluster ~= thruthVals);
end

hold on
plot(subcenters(cluster,1),subcenters(cluster,2),'.b','MarkerSize',15,'LineWidth',2)
plot(subcenters(~cluster,1),subcenters(~cluster,2),'xr','MarkerSize',15,'LineWidth',2)

missClassifieds = (cluster ~= thruthVals);
plot(subcenters(missClassifieds,1),subcenters(missClassifieds,2),'og','MarkerSize',15,'LineWidth',3) % errors
hold off

set(gca,'ydir','normal');
title(totalError);

%% Plot Checker
checkImg = flipud(checkMap);
[rs, cs] = size(cimg);

if (subcenters(1) > cs*0.5)
   checkImg = fliplr(checkImg);
end

[rs, cs] = size(checkImg);
for i=1:rs
    for j=1:cs
        indx = checkImg(i,j);
        if indx ~= -1
           checkImg(i,j) = cluster(indx); 
        else
            checkImg(i,j) = 0.5;
        end
    end
end

figure;
heatmap(checkImg);

figure;
heatmap(normxcorr2(checkImg, checkImg));

%% Function Scope
function r = withinBoundary(b,p)
    p = round(p);

    r = true;
    if p(1) > b(1)
        r = false;
    end

    if p(2) > b(2)
        r = false;
    end

    if p(1) < 1
        r = false;
    end

    if p(2) < 1
        r = false;
    end
end

function [tor, toc] = covariability(img, d)
    tor = 0;
    toc = 0;
    [rs, cs] = size(img);
    for y=1:rs
        for x=1:(cs-d)
           tor = tor + (img(y,x) - img(y,x+d));
        end
    end
    tor = tor * (1/(rs*(cs-d)));
    
    for x=1:cs
        for y=1:(rs-d)
           toc = toc + (img(y,x) - img(y+d,x));
        end
    end
    toc = toc * (1/(cs*(rs-d)));
end

function stats = appGlcm(img)
    [glcm, simg] = graycomatrix(img,'NumLevels',8,'offset',[0 1; -1 0; 0 2; -2 0; 0 3; -3 0; 0 4; -4 0; 0 5; -5 0; 0 6; -6 0; 0 7; -7 0; 0 8; -8 0; 0 9; -9 0; 0 10; -10 0;],'Symmetric',true);
    stats = GLCM_Features(glcm);
    
    cov = zeros(10,2);
    parfor i=1:10
        [r, c] = covariability(simg,i);
        cov(i,:) = [r c];
    end
    cov = reshape(cov, 1, []);
    
    stats = [stats.contr, stats.corrp, stats.energ, stats.homop, stats.entro, cov];
end

function corners = intersectLines(line1, line2, line3, line4)
    [p1x, p1y] = polyxpoly(line1(1,:),line1(2,:),line3(1,:),line3(2,:));
    [p2x, p2y] = polyxpoly(line1(1,:),line1(2,:),line4(1,:),line4(2,:));
    [p3x, p3y] = polyxpoly(line2(1,:),line2(2,:),line3(1,:),line3(2,:));
    [p4x, p4y] = polyxpoly(line2(1,:),line2(2,:),line4(1,:),line4(2,:));
    corners = [p1x,p2x,p3x,p4x;p1y,p2y,p3y,p4y]';
end

function corners = sortULCW(pnts)
    [~, si] = sort(pnts(:,2));
    pnts = pnts(si,:);
    [~, si] = sort(pnts(:,1));
    corners = pnts(si,:);
end

function mapped = mapToRectangle(img, corner)
    h = int64(pdist(corner((1:2),:)));
    w = int64(pdist(corner((2:3),:)));
    
    marg = 0;
    pixSizeW = w + marg * 2;
    pixSizeH = h + marg * 2;
    mapped = zeros(pixSizeH,pixSizeW);
    
    pixSizeW = double(pixSizeW);
    pixSizeH = double(pixSizeH);
    for kk=0:(1/pixSizeW):0.99
        for jj=0:(1/pixSizeH):0.99
            lup = int64(blinMap([jj,kk], corner));
            yk = int64(kk*pixSizeW+1);
            xj = int64(jj*pixSizeH+1);
            mapped(xj, yk) = img(lup(2), lup(1));
        end
    end
end