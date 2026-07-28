function [batch, filesTable] = init_batch()

%filesTable = cell2table(cell(0,3));
%filesTable.Properties.VariableNames =        {'FileName','PathName','frameCycleTime'};
filesTable = cell2table(cell(0,2));
filesTable.Properties.VariableNames =        {'FileName', 'frameCycleTime'};

batch = struct;

%Struct that contains all results
batch.results = struct(...
    'spotsAll',                 {[]},...            %Cell array containing detected spots. One frame per cell (see find_spots_wavelet.m)
    'tracks',                   {[]},...            %Cell array containing tracked molecules >= minTrackLength. One track per cell. (see nearest_neighbour.m and uTrack_wrapper.m)
    'nonLinkedSpots',           [],...              %Array containing spots that have not been linked through tracking
    'tracksSubRoi',             [],...              %Array indicating the subROI number of each track for tracks >= minTrackLength.
    'nonLinkedSpotsSubRoi',     [],...              %Array indicating the subROI number of each non-linked spot
    'trackLengths',             [],...              %Array containing the amount of frames of each track
    'startEndFrameOfTracks',    [],...              %Array containing the first and last frame a track appears
    'meanTrackLength',          0,...               %Array containing the mean Tracklength for tracks 
    'nSpots',                   0,...               %Amount of detected spots
    'nTracks',                  0,...               %Amount of tracks >= minTrackLength
    'nNonLinkedSpots',          0,...               %Amount of non-linked detected spots
    'trackedFraction',          0,...               %Fraction of spots which have been connected to tracks
    'roiSize',                  0,...               %Amount of Pixels within the Region of Interest
    'meanTracksPerFrame',       0,...               %Average number of tracks in one frame
    'meanSpotsPerFrame',        0,...               %Average number of spots in one frame
    'trackDensity',             0,...               %meanTracksPerFrame divided by roiSize
    'jumpDistances',            {[]},...            %Cell array containing the jump Distances between spots of a track for tracks
    'meanJumpDists',            [],...              %Array containing the mean jump Distances of tracks
    'angles',                   {[]},...            %Cell array containing the angles between track segments of tracks
    'nFramesAnalyzed',          0,...               %Number of analyzed frames
    'nSubRegions',              0,...               %Amount of subregions
    'subRegionResults',         struct);            %Structure array containing the results in each subRegion

%Struct that contains all paramteres for detection and tracking (see tracking_routine.m)
batch.params = struct( ...
    'gaussFitWidth',            1.2, ...            %Area used for fitting in fit_spots.m is windowSize*2-1 attention
    'minWidth',                 0, ...              %Minimum Variance in x -and y-direction of Gaussian -> if smaller than minWidth, spot is discarted (currently not used)
    'maxWidth',                 inf, ...            %Maximum variance x -and y-direction of Gaussian -> if bigger than maxWidth, spot is discarted (currently not used)
    'maxRefinementDist',        2,...               %Maximum distance which is accepted for spot refinement
    'minSpotDist',              3.5,...               %Minimum distance two spots have to be appart, if closer together one of the two spots gets rejected attention
    'thresholdFactor',          NaN,...             %Threshold facator which is used to calculate intThreshold
    'trackingRadius',           3,...               %Maximum allowed pixel range for tracking
    'minTrackLength',           20,...              %Minimum track length in frames to count as track
    'gapFrames',                NaN,...             %Amount of frames a spot is allowed to dissapear to still be part of a track
    'minLengthBeforeGap',       NaN,...             %Minimum amount of frames a track has to exist before a gap frame "jump" is allowed
    'frameRange',               [],...              %Framerange for evaluation
    'stdFiltered',              NaN,...             %Standard deviation of filtered stack
    'intThreshold',             NaN,...             %Intensity threshold calculated through standard deviation of original movie * SNR
    'trackingMethod',           NaN,...             %Wether to use nearest neighbour or utrack
    'utrackMotionType',         0,...               %utrack Motion Type, 0 1 2
    'subRoiBorderHandling',     '',...              %Define how tracks crossing sub-region borders should be handled
    'version',                  '',...              %Version of the TrackIt software
    'timeStamp',                datetime('now'));   

%Struct that conatins all the movie informations
batch.movieInfo = struct(...
    'height',               0,...                   %Number of pixels in vertical direction
    'width',                0,...                   %Number of pixels in horizontal direction
    'frames',               0,...                   %Number of frames
    'pathName',             '',...                  %Path where movie file was loaded from
    'fileName',             '',...                  %Name of file where movie was loaded from
    'frameCycleTime',       -1,...                  %Time between the beginning of two consecutive frames: frameCycleTime = camera exposure time + waiting time
    'pathName2',            '',...                  %Path where second movie/image file was loaded from
    'fileName2',            '');                    %Name of file where second movie/image was loaded from

batch.ROI =                 {};                     %Cell array where the first cell contains a list of x -and y-points defining a region of interest
batch.subROI =              {};                     %Cell array where each cell contains one list of x -and y-points per subregion (see assign_tracks_to_regions.m and )

end
