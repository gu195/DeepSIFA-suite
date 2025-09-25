function results = creatResults(spotsAll, tracksAll, ROI, subROI, minTrackLength, frameRange, subRoiBorderHandling)


%% results = create_results(spotsAll, tracksAll, ROI, subROI, minTrackLength, frameRange, subRoiBorderHandling)
%
% Use the raw information of all detected spots and tracks to create the
% results variable that is part of the batch structure. Tracks and spots
% are assigned to the sub-regions (if there are any). Tracks that are
% shorter than the minimum track length are discarded and their detections
% are saved as non-linked spots. Basic statistics eg. jump distances, 
% track lengths, tracked fraction etc. are directly calculated and saved in
% the results variable.
% 
%
% Input:
%   spotsAll	-   Cell array that has as many cells as there are frames  
%                   in the image stack. Each cell contains an array with 8
%                   columns [xpos,ypos,A,BG,sigma_x,sigma_y,angle,exitflag]
%                       1: x-position of the fitted spots
%                       2: y-position of the fitted spots
%                       3: A - maximum of the fitted gaussian
%                       4: BG - Background level of the gaussian
%                       5: sigma_x - variance of the gaussian in x-direction
%                       6: sigma_y - variance of the gaussian in y-direction
%                          (= -1 because isotropic gaussian is fitted)
%                       7: angle - angle at which an anisotropic gaussian
%                          is fitted (= 0 because isotropic gaussian is fitted)
%                       8: exitflag - States if fitting was successfull
% 
%   tracksAll   -   Cell array that has as many entries as there are
%                   tracks. Each cell contains an array with 9 columns:
%                   [frame,xpos,ypos,A,BG,sigma_x,sigma_y,angle,exitflag]
%                   The first column contains the frame in which the spot
%                   appears, the other columns are equal to those in the
%                   spotsAll variable.
%   ROI         -   Single cell containing a 2-column vector with  
%                   coordinates of a region of interest where the spots 
%                   have been detected. 1st column x-coordinates 
%                   (horizontal),2nd column y-coordinates (vertical).
%   subROI      -   Cell array containing one sub-region per cell. Each
%                   sub-region cell contains one cell per frame which again 
%                   contains one cell per region part(one sub-region can be
%                   composed of several separate regions). The coordinates
%                   are stored in a 2-column vector. If the region
%                   was drawn by hand, the region is saved in the first
%                   cell.
%                   Example: if you draw a single sub-region by hand it
%                   would be saved in subROI{1}{1}{1} because it is the
%                   first sub-region in the first frame and consists only
%                   of one part. If you draw another sub-region by hand it
%                   would be saved in subROI{2}{1}{1}. If the two
%                   sub-regions are merged so that the two drawn regions
%                   belong to the same sub-region number, they would be saved
%                   in subROI{1}{1}{1} and subROI{1}{1}{2}. If a sub-region
%                   is created using the "threshold" button, the sub-region
%                   is drawn for each frame. So subROI{1}{5}{2} would be
%                   the first sub-region in the 5th frame and the second
%                   part of multiple regions in this frame.
%   minTrackLength                     
%               -   Minimum number of frames a molecule has to persist 
%                   to be accepted as a track. 
%   frameRange  -   Frames in between which the spots should be detected 
%                   specified as an array with two entries eg. [1 1000]
%   subRoiBorderHandling
%           	-   Defines how to proceed with tracks that cross
%                  	sub-region borders. Pass an empty array if not used or
%                   use one of the following:
%                  	'Assign by first appearance' - Tracks are assigned
%                  	to the sub-region where the first detection in the
%                 	track appears in.
%                  	'Split tracks at borders' - Tracks are split at
%                 	sub-region borders and each separate track is then
%                 	assigned to its sub-region.
%                 	'Delete tracks crossing borders' - Delete all
%                  	tracks that thouch a sub-region border.
%                  	'Only use tracks crossing borders' - Use only the
%                 	tracks that touch a sub-region border.
% 
% Output:
%   results     -   Struct that contains the following fields
%                       'spotsAll' - Cell array containing detected spots. One frame per cell
%                       'tracks' - Cell array containing tracked molecules >= minTrackLength. One track per cell.
%                       'nonLinkedSpots' - Array containing spots that have not been linked through tracking
%                       'tracksSubRoi' - Array indicating the subROI number of each track for tracks >= minTrackLength.
%                       'nonLinkedSpotsSubRoi' - Array indicating the subROI number of each non-linked spot
%                       'trackLengths' - Array containing the amount of frames of each track
%                       'startEndFrameOfTracks' - Array containing the first and last frame a track appears
%                       'meanTrackLength' - Array containing the mean Tracklength for tracks 
%                       'nSpots' - Amount of detected spots
%                       'nTracks' - Amount of tracks >= minTrackLength
%                       'nNonLinkedSpots' - Amount of non-linked detected spots
%                       'trackedFraction' - Fraction of spots which have been connected to tracks
%                       'roiSize' - Amount of Pixels within the Region of Interest
%                       'meanTracksPerFrame' - Average number of tracks in one frame
%                       'meanSpotsPerFrame' - Average number of spots in one frame
%                       'trackDensity' - meanTracksPerFrame divided by roiSize
%                       'jumpDistances' - Cell array containing the jump Distances between spots of a track for tracks
%                       'meanJumpDists' - Array containing the mean jump Distances of tracks
%                       'angles' - Cell array containing the angles between track segments of tracks
%                       'nFramesAnalyzed' - Number of analyzed frames
%                       'nSubRegions' - Amount of subregions
%                       'subRegionResults' - Structure array containing the results in each subRegion (same fields as above)


%Get number of subRegions
nSubRegions = length(subROI);

%Check if there are any sub-regions
if nSubRegions ~= 0    
    %Assign tracks to sub-regions and split tracks if necessary    
    [tracksSubRoiAssignment, tracksAll, tracksDeleted] = assign_tracks_to_regions(subROI, tracksAll, subRoiBorderHandling);
else
    tracksSubRoiAssignment = [];
    tracksDeleted = [];
end

%Calculate the track durations and get the first and last frame of each track
trackLengths = zeros(length(tracksAll),1);
startEndFrameOfTracks =zeros(length(tracksAll),2);

for trackIdx=1:length(tracksAll)
    trackLengths(trackIdx) = (tracksAll{trackIdx}(end,1)-tracksAll{trackIdx}(1,1)+1);
    startEndFrameOfTracks(trackIdx,:) = [tracksAll{trackIdx}(1,1) tracksAll{trackIdx}(end,1)];
end

%Get tracks shorter than shortestTrack and combine them with tracks deleted
%in the sub-region assignement process
tracksTooShort = [tracksAll(trackLengths < minTrackLength) tracksDeleted];

if isempty(tracksTooShort)
    nonLinkedSpots = [];
else
    %Save all tracks that are not used in a combined variable
    nonLinkedSpots = vertcat(tracksTooShort{:});
    
    %Sort non-linked spots by frame
    [~,idx] = sort(nonLinkedSpots(:,1));
    nonLinkedSpots = nonLinkedSpots(idx,:);
end

%Get the indices of tracks which have a minimum length of minTrackLength
trackFilter = trackLengths >= minTrackLength;

%Use only tracks which have a minimum length of minTrackLength
tracks = tracksAll(trackFilter);
trackLengths = trackLengths(trackFilter);
startEndFrameOfTracks = startEndFrameOfTracks(trackFilter,:);

if ~isempty(tracksSubRoiAssignment)
    tracksSubRoiAssignment = tracksSubRoiAssignment(trackFilter);
end


%Calculate average binding time. Set to zero if no tracks exist.
meanTrackLength = max(0, mean(trackLengths));

%Count all spots
nSpots = 0;

for frameIdx=1:length(spotsAll)
    nSpots = nSpots + size(spotsAll{frameIdx},1);
end

%Count number of tracks
nTracks = length(tracks);

%Count number of non-linked spots
nNonLinkedSpots = size(nonLinkedSpots,1);

%Calculate fraction of spots which have been connected to tracks
trackedFraction = nTracks/(nTracks+nNonLinkedSpots);

%Calculate ROI size
roiSize = polyarea(ROI{1}(:,1), ROI{1}(:,2));

%Get number of analyzed frames
nFramesAnalyzed = frameRange(2)-frameRange(1)+1;

%Get jumpDistances, angles and number of spots in tracks
meanJumpDists = zeros(nTracks,1);
jumpDistances = cell(nTracks,1);
angles = cell(nTracks,1);
nSpotsInTracks = 0;

for k = 1:length(tracks)
    x = tracks{k}(:,2);
    y = tracks{k}(:,3);
    xSqDisp = (x(1:end-1) - x(2:end)).^2;
    ySqDisp = (y(1:end-1) - y(2:end)).^2;
    
    meanJumpDists(k) = mean(sqrt(xSqDisp + ySqDisp));
    jumpDistances{k} = sqrt(xSqDisp + ySqDisp);
    angles{k} = getAngles(tracks{k}(:,2:3));
    
    nSpotsInTracks = nSpotsInTracks + numel(x);
end

%Calculate mean number of tracks and spots per frame
meanTracksPerFrame = nSpotsInTracks/nFramesAnalyzed;
meanSpotsPerFrame = nSpots/nFramesAnalyzed;

%Calculate track density
trackDensity = meanTracksPerFrame/roiSize;

%Save results
results.spotsAll = spotsAll;
results.nonLinkedSpots = nonLinkedSpots;
results.tracks = tracks;

results.nSpots = nSpots;
results.nTracks = nTracks;
results.nNonLinkedSpots = nNonLinkedSpots;
results.trackedFraction = trackedFraction;

results.trackLengths = trackLengths;
results.startEndFrameOfTracks = startEndFrameOfTracks;
results.jumpDistances = jumpDistances;
results.meanJumpDists = meanJumpDists;
results.angles = angles;
results.meanTracksPerFrame = meanTracksPerFrame;
results.meanSpotsPerFrame = meanSpotsPerFrame;
results.meanTrackLength = meanTrackLength;
results.roiSize = roiSize;
results.trackDensity = trackDensity;
results.nFramesAnalyzed = nFramesAnalyzed;

if nSubRegions ~= 0
    %Assign non-linked Spots to regions
    nonLinkedSpotsSubRoiAssignment = assign_spots_to_regions(subROI, nonLinkedSpots);
    %Create results for each sub region
    subRegionResults = create_subregion_results(results, nFramesAnalyzed, subROI, tracksSubRoiAssignment, nonLinkedSpotsSubRoiAssignment);
else
    subRegionResults = [];
    nonLinkedSpotsSubRoiAssignment = [];
end

results.nSubRegions = nSubRegions;
results.tracksSubRoi = tracksSubRoiAssignment;
results.nonLinkedSpotsSubRoi = nonLinkedSpotsSubRoiAssignment;
results.subRegionResults = subRegionResults;
end


function subRegionResults = create_subregion_results(results, nFramesAnalyzed, subRoi, tracksSubRoiAssignment, nonLinkedSpotsSubRoiAssignment)

tracks = results.tracks;
startEndFrameOfTracks = results.startEndFrameOfTracks;
trackLengths = results.trackLengths;
meanJumpDists = results.meanJumpDists;
jumpDistances = results.jumpDistances;
angles = results.angles;

%--------%Create results for sub-regions-----------------------------------

subRegionResults = struct('name','','nSpots',0,'nTracks',0,'nNonLinkedSpots',0,...
    'trackLengths',[],'jumpDistances',{[]},'meanJumpDists',[],'angles',{[]},...
    'meanTracksPerFrame',0,'meanSpotsPerFrame',0,'roiSize',0,'trackDensity',0,...
    'meanTrackLength',0,'trackedFraction',0,'startEndFrameOfTracks',[]);

%Initialize subRegion Results for main region + sub-regions (main region on position 1)
nSubRegions = length(subRoi);
subRegionResults = repmat(subRegionResults,nSubRegions+1,1);

for subRegionIdx = 1:nSubRegions+1
    
    %Create region name
    if subRegionIdx == 1
        subRegionResults(subRegionIdx).name = ['Region ', num2str(subRegionIdx), ' (tracking-region)'];
    else
        subRegionResults(subRegionIdx).name = ['Region ' num2str(subRegionIdx)];
    end
    
    %Get logical indices of tracks in this region
    tracksInCurRegionIdx = (subRegionIdx-1) == tracksSubRoiAssignment;
    
    %Get number of non-linked spots in this region
    nNonLinkedSpots  = sum((subRegionIdx-1) == nonLinkedSpotsSubRoiAssignment);
    subRegionResults(subRegionIdx).nNonLinkedSpots = nNonLinkedSpots;
    
    %Get number of tracks in this region
    nTracks = sum(tracksInCurRegionIdx);
    subRegionResults(subRegionIdx).nTracks = nTracks;
    
    %Calculate region sizes
    if subRegionIdx == 1
        subRegionResults(subRegionIdx).roiSize = results.roiSize;
    else
        curSubRegion = subRoi{subRegionIdx-1};
        nFrameParts = length(curSubRegion);
        
        subRoiSizePerFrame = zeros(nFrameParts,1);
        
        for frameIdx=1:nFrameParts
            areaSizeCurFrame = 0;
            for n =1:length(curSubRegion{frameIdx}) %Iterate through all parts of boundary
                areaSizeCurFrame = areaSizeCurFrame + polyarea(curSubRegion{frameIdx}{n}(:,1),curSubRegion{frameIdx}{n}(:,2));
            end
            
            subRoiSizePerFrame(frameIdx) = areaSizeCurFrame ;
        end
        
        subRegionResults(subRegionIdx).roiSize = mean(subRoiSizePerFrame);
    end
    
    if subRegionResults(subRegionIdx).nTracks ~= 0
        %Get tracklengths
        trackLengthsCurRegion = trackLengths(tracksInCurRegionIdx);
        subRegionResults(subRegionIdx).trackLengths = trackLengthsCurRegion;
        
        %Calculate mean tracklenght
        subRegionResults(subRegionIdx).meanTrackLength = mean(trackLengthsCurRegion);
        
        %Get first and last frame of tracks in this region
        subRegionResults(subRegionIdx).startEndFrameOfTracks = startEndFrameOfTracks(tracksInCurRegionIdx,:);
        %Get displacements
        subRegionResults(subRegionIdx).jumpDistances = jumpDistances(tracksInCurRegionIdx);
        %Get mean displacements
        subRegionResults(subRegionIdx).meanJumpDists = meanJumpDists(tracksInCurRegionIdx);
        %Get jump angles
        subRegionResults(subRegionIdx).angles = angles(tracksInCurRegionIdx);
        
        %Calculate fraction of spots which have been connected to tracks
        trackedFraction = nTracks/(nTracks+nNonLinkedSpots);
        
        %Get total number of spots in this region
        nSpotsInTracks = 0;
        
        for trackIdx = find(tracksInCurRegionIdx)'
            nSpotsInTracks = nSpotsInTracks + size(tracks{trackIdx},1);
        end
        
        nSpots = nSpotsInTracks + nNonLinkedSpots;
        subRegionResults(subRegionIdx).nSpots = nSpots;
        subRegionResults(subRegionIdx).trackedFraction = trackedFraction;
        
        %Calculate average number of tracks per frame
        subRegionResults(subRegionIdx).meanTracksPerFrame = nSpotsInTracks/nFramesAnalyzed;

        %Calculate average number of spots per frame
        subRegionResults(subRegionIdx).meanSpotsPerFrame = nSpots/nFramesAnalyzed;
        
                
        %Calculate track density
        subRegionResults(subRegionIdx).trackDensity = subRegionResults(subRegionIdx).meanTracksPerFrame/subRegionResults(subRegionIdx).roiSize;
        
    end
    
    
end
end

