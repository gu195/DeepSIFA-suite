function tracks = nearNeighbour(spots,trackingRadius,gapFrames, minLengthBeforeGap,ui)

%%This function is used to connect detected spots in a single-molecule movie
%into tracks using a nearest neighbour algorithm. It uses the freely
%available nearest nighbor linker from Jean-Yves Tinevez (2022):
%Nearest-neighbor linker (https://www.mathworks.com/matlabcentral/fileexchange/33772-nearest-neighbor-linker)
%
%
% tracks = nearest_neighbour(spots,trackingRadius,gapFrames, minLengthBeforeGap);
%
%
%
%Input:
%   spots               -   Cell array that has as many entries as there are
%                           frames in the image stack. Each cell contains an
%                           array with at least 2 columns with the x-coordinates
%                           in the 1st column and y-coordinates in the 2nd column.
%                           Eg. spots{10} = [5.1,10.2; 20.5,30.1] indicates that
%                           there are two spots, one at (x=5.1, y=10.2) and one
%                           at(x=20.5, y=30.1) in the 10th frame of the stack.
%   trackingRadius      -   Maximum allowed distance to connect spots into tracks
%   gapFrames           -   Amount of frames a molecule is allowed to dissapear to
%                           still be connected into a track.
%   minLengthBeforeGap  -   Minimum amount of frames a track has to exist
%                           before a connection over a gap frame is allowed.
%
%Output:
%   tracks              -   Cell array that has as many entries as there
%                           are tracks. Each cell contains an array with at
%                           least 3 columns [frame,xpos,ypos]. If spot
%                           positions were refined with the fit_spots
%                           function each cell contains an array with 9 columns:
%                           [frame,xpos,ypos,A,BG,sigma_x,sigma_y,angle,exitflag]
%                               1: frame in which spot appears
%                               2: x-position of the fitted spots
%                               3: y-position of the fitted spots
%                               4: A - maximum of the fitted gaussian
%                               5: BG - Background level of the gaussian
%                               6: sigma_x - variance of the gaussian in x-direction
%                               7: sigma_y - variance of the gaussian in y-direction
%                                   (= -1 because isotropic gaussian is fitted)
%                               8: angle - angle at which an anisotropic gaussian
%                                   is fitted (= 0 because isotropic gaussian is fitted)
%                               9: exitflag - States if fitting was successfull
%
%
%Example:
%   tracks{1} = [1, 5.1, 10.2, 1044.3, 924.7, 1.3, -1, 0, 1;
%                2, 7.2,  9.2, 1030.1, 910.5, 1.2, -1, 0, 1;
%                4, 5.5, 30.1, 1050.9, 830.2, 1.4, -1, 0, 1]]
%   This would be the first track that has been found. The spots connected
%   to the track appear in the 1st, 2nd and 4th frame. The 3rd frame is a
%   "gap frame".

if nargin == 4
    ui.editFeedbackWin.String = '';
end

origFeedbackWin = ui.editFeedbackWin.String(2:end,:);

tracks          = {};
nFrames = length(spots);
activeTrails    = zeros(0, 4); % index, last frame, last x, last y

for frameIdx = 1:nFrames %Iterate through frames
    
    spotsSource = activeTrails(:,3:4);
    spotsTarget = spots{frameIdx}(:,1:2);
    
    nSourcePoints = size(spotsSource, 1);
    nTargetPoints = size(spotsTarget, 1);
        
    if nTargetPoints>0
        D = NaN(nSourcePoints, nTargetPoints);
        
        % Build distance matrix
        for i = 1 : nSourcePoints
            
            % Pick one source point
            curPoint = spotsSource(i, :);
            
            % Compute square distance to all target points
            diffCords = spotsTarget - repmat(curPoint, nTargetPoints, 1);
            squareDists = sum(diffCords.^2, 2);
            
            % Store them
            D(i, :) = squareDists;
            
        end
        
        % Deal with maximal linking distance: we simply mark these links as already
        % treated, so that they can never generate a link.
        
        D (D > trackingRadius^2) = Inf;
        
        targetIndices = -1 * ones(nSourcePoints, 1);
        targetDistances = NaN(nSourcePoints, 1);
        
        
        % Parse distance matrix
        while ~all(isinf(D(:)))
            
            [ min_D, closestTargets ] = min(D, [], 2); % index of the closest target for each source points
            [ ~, sortedIndex ] = sort(min_D);
            
            for i = 1 : numel(sortedIndex)
                sourceIndex =  sortedIndex(i);
                targetIndex =  closestTargets (sortedIndex(i));
                
                % Did we already assigned this target to a source?
                if any (targetIndex == targetIndices) || min_D(sourceIndex) == inf
                    
                    % Yes, then exit the loop and change the distance matrix to
                    % prevent this assignment
                    break
                    
                else
                    
                    % No, then store this assignment
                    targetIndices( sourceIndex ) = targetIndex;
                    targetDistances ( sourceIndex ) = sqrt ( min_D (  sortedIndex(i) ) );
                    
                    % And make it impossible to find it again by putting the target
                    % point to infinity in the distance matrix
                    D(:, targetIndex) = Inf;
                    % And the same for the source line
                    D(sourceIndex, :) = Inf;
                    
                    activeTrails(sourceIndex,2:4) = [frameIdx, spots{frameIdx}(targetIndex, 1:2)];
                    trailIndex = activeTrails(sourceIndex,1);
                    tracks{trailIndex}  = [tracks{trailIndex}; [frameIdx, spots{frameIdx}(targetIndex, :)]];
                end
                
            end
            
        end
        
        unassignedTargets = setdiff(1 : nTargetPoints , targetIndices);
        nUnassignedTargets = numel(unassignedTargets);
        
        newTrailIndices = length(tracks)+1:length(tracks)+nUnassignedTargets;
        newTrails    = [ones(nUnassignedTargets,1)*(frameIdx), spots{frameIdx}(unassignedTargets, :)]; %Contains frame# in first column // One row for each spot (which did not match)
        
        tracks(newTrailIndices) = mat2cell(newTrails, ones(size(newTrails, 1), 1), size(newTrails, 2)); %Add cells containing new spots (which did not match) at the end of the cell array
        activeTrails = [activeTrails;[newTrailIndices',newTrails(:,1:3)]];
    end
    
    activeTrails = activeTrails(frameIdx-activeTrails(:, 2) <= gapFrames, :); %Throw out spots which have been dark longer than gapFrames
    
    %Close tracks which are shorter than the minimum segmentation length so
    %they don't get connected via gapFrames
    
    if minLengthBeforeGap > 1
        indicesOfTrailsWithgapFrames = frameIdx-activeTrails(:, 2) > 0;
        
        [trackLengths,~] = cellfun(@size, tracks(activeTrails(:,1)),'UniformOutput', false);
        indicesOfTrailsShorterMinSeg = transpose(cell2mat(trackLengths) < minLengthBeforeGap);
        indicesCombined = indicesOfTrailsWithgapFrames & indicesOfTrailsShorterMinSeg;
        activeTrails = activeTrails(~indicesCombined,:);
    end
    
    percentDone = round(frameIdx * 100 / nFrames);
    if mod(percentDone,1) == 0
        msg = sprintf('Finding tracks: %3.0f %%', percentDone);
        ui.editFeedbackWin.String = char(msg, origFeedbackWin);
        drawnow
        %if double(get(gcf,'CurrentCharacter')) == 24
        %    break
        %end
    end
    
    
end

end