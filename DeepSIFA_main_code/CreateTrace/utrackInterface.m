function tracks = utrackInterface(spots, trackingRadius, gapFrames,minLengthBeforeGap, linearMotion)

%将所有帧的亮斑输入给u-track, 通过u-track删除无关的独立光斑，只保留连续光斑
%
%Input:
%   spots               -   cell array形式的光斑信息，包含坐标
%                           比如spots{10} = [5.2,8.3; 101.6,150.1] 表示第十帧有两个光斑
%   trackingRadius      -   每帧最大移动距离px
%   gapFrames           -   允许消失的最大帧数
%   minLengthBeforeGap  -   消失前至少存在的帧数
%   motionType          -   1：u-track linear motion + brownian
%                           0: brownian motion model
%                           
%Output:
%   tracks              -   光斑轨迹，Cell array，至少包含[frame,xpos,ypos]
%                               1: frame 
%                               2: x-coord
%                               3: y-coord
%                               4: 高斯拟合幅值
%
%
%                               tracks{1} = [1, 5.1, 10.2, 1044.3;
%                                            2, 7.2,  9.2, 1030.1;
%                                            4, 5.5, 30.1, 1050.9]

nFramesAnalyzed = length(spots);
%spotsUTrack = repmat(struct('xCoord',[],'yCoord',[],'amp',[]),nFramesAnalyzed,1);
spotsUTrack = repmat(struct('xCoord',[],'yCoord',[],'amp',[],...
                            'BG',[],'sigmax',[],'sigmay',[]),nFramesAnalyzed,1);

for m = 1:nFramesAnalyzed
    if~isempty(spots{m})
        nCand = size(spots{m},1);
        spotsUTrack(m).xCoord = [spots{m}(:,1) zeros(nCand,1)];     % x-coordinate 
        spotsUTrack(m).yCoord = [spots{m}(:,2) zeros(nCand,1)];     % y-coordinate 
        spotsUTrack(m).amp    = [spots{m}(:,3) zeros(nCand,1)];     % A
        spotsUTrack(m).BG     = [spots{m}(:,4) zeros(nCand,1)];     % BG
        spotsUTrack(m).sigmax = [spots{m}(:,5) zeros(nCand,1)];     % sigma_x
        spotsUTrack(m).sigmay = [spots{m}(:,6) zeros(nCand,1)];     % singma_y   
    end
end

spotsUTrack2 = repmat(struct('xCoord',[],'yCoord',[],'amp',[]),nFramesAnalyzed,1);

for m = 1:nFramesAnalyzed
    if~isempty(spots{m})
        nCand = size(spots{m},1);
        spotsUTrack2(m).xCoord = [spots{m}(:,1) zeros(nCand,1)];     % x-coordinate without using standard deviation of fit
        %         spotsUTrack(m).xCoord = [spots{m}(:,1) spots{m}(:,5)];        % x-coordinate plus standard deviation of fit
        spotsUTrack2(m).yCoord = [spots{m}(:,2) zeros(nCand,1)];     % y-coordinate without using standard deviation of fit
        %         spotsUTrack(m).yCoord = [spots{m}(:,2) spots{m}(:,6)];      % y-coordinate plus standard deviation of fit
        spotsUTrack2(m).amp    = [spots{m}(:,3) zeros(nCand,1)];     % Fitted spot intensity
    end
end

%% ----------以下部分改自u-track/software scriptTrackGeneral.m -----------

gapCloseParam.timeWindow = gapFrames+1; %maximum allowed time gap (in frames) between a track segment end and a track segment start that allows linking them.
gapCloseParam.mergeSplit = 1; %1 if merging and splitting are to be considered, 2 if only merging is to be considered, 3 if only splitting is to be considered, 0 if no merging or splitting are to be considered.
gapCloseParam.minTrackLen = minLengthBeforeGap; %minimum length of track segments from linking to be used in gap closing.
gapCloseParam.diagnostics = 0; %1 to plot a histogram of gap lengths in the end; 0 or empty otherwise.
%% cost matrix for frame-to-frame linking

%function name
costMatrices(1).funcName = 'costMatRandomDirectedSwitchingMotionLink';

%parameters

parameters.linearMotion = linearMotion; %use linear motion Kalman filter.
parameters.minSearchRadius = 0; %minimum allowed search radius. The search radius is calculated on the spot in the code given a feature's motion parameters. If it happens to be smaller than this minimum, it will be increased to the minimum.
parameters.maxSearchRadius = trackingRadius; %maximum allowed search radius. Again, if a feature's calculated search radius is larger than this maximum, it will be reduced to this maximum.
parameters.brownStdMult = 3; %multiplication factor to calculate search radius from standard deviation.

parameters.useLocalDensity = 1; %1 if you want to expand the search radius of isolated features in the linking (initial tracking) step.
parameters.nnWindow = gapCloseParam.timeWindow; %number of frames before the current one where you want to look to see a feature's nearest neighbor in order to decide how isolated it is (in the initial linking step).

parameters.kalmanInitParam = []; %Kalman filter initialization parameters.
% parameters.kalmanInitParam.searchRadiusFirstIteration = 10; %Kalman filter initialization parameters.

%optional input
parameters.diagnostics = []; %if you want to plot the histogram of linking distances up to certain frames, indicate their numbers; 0 or empty otherwise. Does not work for the first or last frame of a movie.

costMatrices(1).parameters = parameters;
clear parameters

%% cost matrix for gap closing

%function name
costMatrices(2).funcName = 'costMatRandomDirectedSwitchingMotionCloseGaps';

%parameters

%needed all the time
parameters.linearMotion = linearMotion; %use linear motion Kalman filter.

parameters.minSearchRadius = 0; %minimum allowed search radius.
parameters.maxSearchRadius = trackingRadius; %maximum allowed search radius.
parameters.brownStdMult = 3*ones(gapCloseParam.timeWindow,1); %multiplication factor to calculate Brownian search radius from standard deviation.

parameters.brownScaling = [0.25 0.01]; %power for scaling the Brownian search radius with time, before and after timeReachConfB (next parameter).
% parameters.timeReachConfB = 3; %before timeReachConfB, the search radius grows with time with the power in brownScaling(1); after timeReachConfB it grows with the power in brownScaling(2).
parameters.timeReachConfB = gapCloseParam.timeWindow; %before timeReachConfB, the search radius grows with time with the power in brownScaling(1); after timeReachConfB it grows with the power in brownScaling(2).

parameters.ampRatioLimit = []; %for merging and splitting. Minimum and maximum ratios between the intensity of a feature after merging/before splitting and the sum of the intensities of the 2 features that merge/split.

parameters.lenForClassify = 5; %minimum track segment length to classify it as linear or random.

parameters.useLocalDensity = 0; %1 if you want to expand the search radius of isolated features in the gap closing and merging/splitting step.
parameters.nnWindow = gapCloseParam.timeWindow; %number of frames before/after the current one where you want to look for a track's nearest neighbor at its end/start (in the gap closing step).

parameters.linStdMult = 3*ones(gapCloseParam.timeWindow,1); %multiplication factor to calculate linear search radius from standard deviation.

parameters.linScaling = [1 0.01]; %power for scaling the linear search radius with time (similar to brownScaling).
% parameters.timeReachConfL = 4; %similar to timeReachConfB, but for the linear part of the motion.
parameters.timeReachConfL = gapCloseParam.timeWindow; %similar to timeReachConfB, but for the linear part of the motion.

parameters.maxAngleVV = 30; %maximum angle between the directions of motion of two tracks that allows linking them (and thus closing a gap). Think of it as the equivalent of a searchRadius but for angles.

%optional; if not input, 1 will be used (i.e. no penalty)
parameters.gapPenalty = 1.5; %penalty for increasing temporary disappearance time (disappearing for n frames gets a penalty of gapPenalty^(n-1)).

%optional; to calculate MS search radius
%if not input, MS search radius will be the same as gap closing search radius
parameters.resLimit = []; %resolution limit, which is generally equal to 3 * point spread function sigma.

%NEW PARAMETER
parameters.gapExcludeMS = 1; %flag to allow gaps to exclude merges and splits

%NEW PARAMETER
parameters.strategyBD = -1; %strategy to calculate birth and death cost

costMatrices(2).parameters = parameters;
clear parameters

%% Kalman filter function names

kalmanFunctions.reserveMem  = 'kalmanResMemLM';
kalmanFunctions.initialize  = 'kalmanInitLinearMotion';
kalmanFunctions.calcGain    = 'kalmanGainLinearMotion';
kalmanFunctions.timeReverse = 'kalmanReverseLinearMotion';

%verbose state
verbose = 1;

%problem dimension
probDim = 2;

[trackFinal,~,~,~] = trackCloseGapsKalmanSparse(spotsUTrack,costMatrices,gapCloseParam,kalmanFunctions,probDim,0,verbose);
%% 至此u-track计算结束-------------------

%% 整理结果，使其格式统一

nTracks = numel(trackFinal);
runningTrackId = 1;
tracks = cell(nTracks,1);
%trackFinal(4251).tracksFeatIndxCG(164)
for trackIdx = 1:nTracks
    
    firstFrame = trackFinal(trackIdx).seqOfEvents(1,1); %获取首次出现的帧数
    
    %从u-track结果中构建当前track的struct
    curTrack = [trackFinal(trackIdx).tracksCoordAmpCG(1:8:end)',...
                trackFinal(trackIdx).tracksCoordAmpCG(2:8:end)',...
                trackFinal(trackIdx).tracksCoordAmpCG(4:8:end)'];

    spotIndex = trackFinal(trackIdx).tracksFeatIndxCG;
    
    trackLength2 = size(spotIndex,2);
    trackLength = size(curTrack,1);
    if trackLength < 1
        continue;
    end
    % test: 
    % spotsUTrack(1).xCoord(97, 1)
    % disp('*****************************');disp(trackIdx);
    currentTrack = zeros(trackLength2, 6);  %x, y, A, BG, sig_x, sig_y
    
    for frameIndex = firstFrame:firstFrame+trackLength2-1
        %将frameIndex映射到currentTrack范围1:trackLength
        frameIndexTo1 = frameIndex- firstFrame + 1; 
        k = trackFinal(trackIdx).tracksFeatIndxCG(frameIndexTo1) ; 
        % disp(k);
        if k == 0
            currentTrack(frameIndexTo1, 1:6) = 0;  % 缺帧时以0补充，后删掉
            continue;
        end
        currentTrack(frameIndexTo1, 1) = spotsUTrack(frameIndex).xCoord(k, 1);
        currentTrack(frameIndexTo1, 2) = spotsUTrack(frameIndex).yCoord(k, 1);
        currentTrack(frameIndexTo1, 3) = spotsUTrack(frameIndex).amp(k, 1);
        currentTrack(frameIndexTo1, 4) = spotsUTrack(frameIndex).BG(k, 1);
        currentTrack(frameIndexTo1, 5) = spotsUTrack(frameIndex).sigmax(k, 1);
        currentTrack(frameIndexTo1, 6) = spotsUTrack(frameIndex).sigmay(k, 1);
    end

    frameIndices = firstFrame:firstFrame+trackLength-1;
    frameIndices2 = firstFrame:firstFrame+trackLength2-1;
    
    curTrackWithIdx = [frameIndices', curTrack];
    curTrackWithIdx2 = [frameIndices2', currentTrack];
    
    % Delete gap frames (rows with NaN)
    curTrackWithIdx = curTrackWithIdx(~isnan(curTrackWithIdx(:,end)),:);
    curTrackWithIdx2 = curTrackWithIdx2(~isnan(curTrackWithIdx2(:,end)),:);
    
    %删掉之前缺帧时补的0和nan序列
    spotsUTrack2 = curTrackWithIdx(:,2:3);
    if ~any(isnan(spotsUTrack2(:))) && ~any(spotsUTrack2(:)==0)
        tracks(runningTrackId) = {curTrackWithIdx}; %[frame,x,y,z,amp]
        tracks2(runningTrackId) = {curTrackWithIdx2};
        runningTrackId = runningTrackId + 1;
    end

end

%Delete empty cells
tracks = tracks(~cellfun('isempty', tracks));
tracks2 = tracks2(~cellfun('isempty', tracks2));

%% 保存独立亮斑，即前后无关联的亮斑


allSpots = vertcat(spots{:});
frameNumsOfSpots = ones(size(allSpots,1),1);
runningIdx = 1;

for frameIdx = 1:length(spots)
    if isempty(spots{frameIdx})
        continue
    end
    
    nSpotsInCurFrame = size(spots{frameIdx},1);
    frameNumsOfSpots(runningIdx:runningIdx+nSpotsInCurFrame-1) = frameIdx;
    runningIdx = runningIdx+nSpotsInCurFrame;
end

allSpots = [frameNumsOfSpots, allSpots];
allTracks = vertcat(tracks{:});

inSpotsAndTracksIdx = ismember(allSpots(:,2:3),allTracks(:,2:3));
inSpotsAndTracksIdx = inSpotsAndTracksIdx(:,1)&inSpotsAndTracksIdx(:,2);

nonLinkedSpots = num2cell(allSpots(~inSpotsAndTracksIdx,1:7),2);
tracks = [tracks2, nonLinkedSpots'];
%tracks = tracks';

end