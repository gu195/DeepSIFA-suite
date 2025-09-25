function [batch,filtered] = trackingSM(batch, para, providedStack)


% 单分子荧光追踪主程序%
%
% Input:
%  batch    -   TrackIts batch variable is a struct array
%               containing all relevant informations such as
%               movie infos, results and tracking paramteres.
%               See init_batch.m for detailed description of
%               all fields in a batch file.
%  para     -   Struct with following fields
%                   boolFindSpots: 
%                       Set to true if spots should be detected or false if 
%                       tracking should be performed on existing spots.
%                   thresholdFactor: 
%                       Used to calculate an automatic 
%                       threshold  for spot detection.                       
%                   frameRange: 
%                       Frames in between which the
%                     	spots should be detected specified as a
%                      	2-elemnt array eg. [1 inf]
%                   trackingMethod: 
%                       Algorithm that is used for tracking. Use
%                       Nearest neighbour', 'u-track random motion' 
%                    	or 'u-track linear+random motion'.
%                   tlConditions: 
%                       Vector containing the frame cycle times that appear
%                       in the batch file (leave empty if all movies should 
%                       be analyzed with the same tracking settings
%                       independent of their frame cycle time)
%                   trackingRadius:
%                       Maximum allowed distance to connect spots into
%                       tracks. Vector that contains as many values as
%                       there are values in the field tlConditions.
%                   minTrackLength
%                       Minimum number of frames a molecule has to persist 
%                       to be accepted as a track. Vector that contains as 
%                       many values as there are values in the field t
%                       lConditions.
%                   gapFrames
%                       Amount of frames a molecule is allowed to dissapear 
%                       to still be connected into a track. Vector that 
%                       contains as many values as there are values in the 
%                       field tlConditions.
%                   minLengthBeforeGap attention
%                       Minimum amount of frames a track has to exist 
%                       before a connection over a gap frame is allowed.
%                       Vector that contains as many values as there are 
%                       values in the field tlConditions.
%                   subRoiBorderHandling
%                       String that defines how to proceed with tracks that 
%                       cross sub-region border. Use one of the following:
%                       'Assign by first appearance' - Tracks are assigned
%                       to the sub-region where the first detection in the
%                       track appears in.
%                       'Split tracks at border' - Tracks are split at
%                       sub-region borders and each separate track is then
%                       assigned to its sub-region.
%                       'Delete tracks crossing borders' - Delete all
%                       tracks that thouch a sub-region border.
%                       'Use only tracks crossing borders' - Use only the
%                       tracks that touch a sub-region border.
%                   trackItVersion
%                       String that defines the currently used TrackIt
%                       version
%  providedStack -  If only one movie is to be analyzed, an image
%                  	stack can be passed to the tracking routine.
%                  	This happens if the user presses "Analyze
%                 	current movie" because the movie is already
%                  	loaded. Pass empty array if not used.

% Output:
%     batch         -   Batch structure containing all tracking results.


para.boolFindSpots = 1;
para.tlConditions = [];
%从第0帧开始
para.minLengthBeforeGap = 0;
% 排序方式
para.subRoiBorderHandling = 'Assign by first appearance';

try    
    
    
    nFiles = length(batch);
    
    % Start analysis
    tic
    for n = 1:nFiles
        
        if para.boolFindSpots
            
            %------Prepare image stack, framerange and ROI-----------------
            
            if ~isempty(providedStack) && nFiles == 1
                %Only one movie has to be analyzed and stack was provided
                stack = providedStack;
            else
                %Movie stack was not provided so load stack from file
                stack = tiffLoadStack(batch(n).movieInfo.pathName, batch(n).movieInfo.fileName);
                
                if isempty(stack)
                    disp('Movie not found, check path');
                    return
                end                
                
            end
            
            %Get size of image stack 
            [height, width, nFrames] = size(stack);            
            
                        
            if isempty(batch(n).ROI)
                %Initialize ROI to whole movie if none was drawn
                batch(n).ROI = {[.5 .5; .5 height + .5; width+.5 height+.5; width+.5 .5; .5 .5]};
            end
            
            %Make sure last frame to analyze is not larger than the number of frames in the image stack 
            frameRange = [para.frameRange(1), min(para.frameRange(2),nFrames)];
            
            %--------Find spots--------------------------------------------
            [spots, stdFiltered, intThreshold, filtered] = spotsFind(stack, para.thresholdFactor, frameRange, batch(n).ROI{1});
            
            %--------Fit Spots---------------------------------------------
            [spots, nNonFittedSpots] = spotsFit(stack, spots, batch(n).params);
            
            %--------Save new parameters to the batch structure------------
            batch(n).movieInfo.height = height;
            batch(n).movieInfo.width = width;
            batch(n).movieInfo.frames = nFrames;
                        
            
            batch(n).params.frameRange = frameRange;
            batch(n).params.thresholdFactor = para.thresholdFactor;
            batch(n).params.stdFiltered = stdFiltered;
            batch(n).params.intThreshold = intThreshold;            
            
        else %Use spots from earlier analysis
            spots = batch(n).results.spotsAll;
            frameRange = batch(n).params.frameRange;
            nNonFittedSpots = [];
            
        end
        
        %-------Find Tracks------------------------------------------------
                
        if numel(para.tlConditions) > 1
            %Use time lapse dependent tracking parameters
            curTlNum = para.tlConditions == batch(n).movieInfo.frameCycleTime;
        else
            curTlNum = 1;
        end
        
        trackingRadius = para.trackingRadius(curTlNum);
        gapFrames = para.gapFrames(curTlNum);
        minTrackLength = para.minTrackLength(curTlNum);
        minLengthBeforeGap = para.minLengthBeforeGap(curTlNum);
        %disp('************test***********');
        %tracksAll = nearNeighbour(spots,trackingRadius,gapFrames, minLengthBeforeGap);
        %tracksAll2 = uTrack_wrapper(spots, trackingRadius, gapFrames,minLengthBeforeGap, 0);
        %tracksAll3 = uTrack_wrapper(spots,  trackingRadius, gapFrames, minLengthBeforeGap, 1);
        switch para.trackingMethod
            case {'default', ''}
                tracksAll = nearNeighbour(spots,trackingRadius,gapFrames, minLengthBeforeGap);
            case 'u-track' 
                disp('Starting with u-track');
                motionType = para.utrackMotionType;
                tracksAll = utrackInterface(spots, trackingRadius, gapFrames,minLengthBeforeGap, motionType);
                % motionType = 1 -> movement with constant velocity, 
                % motiontype = 2  movement along a straight line but with the possibility of immediate direction reversal.
                % 参考u-track userguide
                % tracksAll = uTrack_wrapper(spots,  trackingRadius, gapFrames, minLengthBeforeGap, 1);
                disp('Finished tracking with u-track');
        end
        
        %保存参数数据
        batch(n).params.trackingMethod = para.trackingMethod;
        batch(n).params.trackingRadius = trackingRadius;
        batch(n).params.minTrackLength = minTrackLength;
        batch(n).params.gapFrames = gapFrames;
        batch(n).params.minLengthBeforeGap = minLengthBeforeGap;
        batch(n).params.subRoiBorderHandling = para.subRoiBorderHandling;
        
        %-----Create results-----------------------------------------------

        
        results = creatResults(spots, tracksAll, batch(n).ROI, batch(n).subROI, minTrackLength, frameRange, para.subRoiBorderHandling);
        
        %Save results in batch structure
        batch(n).results = results;
        
        %Save timestamp and trackIt version in batch structure
        batch(n).params.timeStamp = datetime('now');
                     
    end
    
    
catch ex
    boolCancelled = 1;
    
    errorMessage = ex.message;
    for idx = 1:numel(ex.stack)
        errorMessage = [errorMessage, sprintf('\nError in function %s() at line %d.', ...
            ex.stack(idx).name, ex.stack(idx).line)];
    end
    
    fprintf(1, '%s\n', errorMessage,'')

end




end





