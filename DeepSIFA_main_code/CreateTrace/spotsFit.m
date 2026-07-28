function [spotsFitted, nNonfittedSpots] = spotsFit(stack, spots, params)


%Refine positions of detected spots in a single-molecule movie by fitting
%an isotropic gaussian point spread function to the pixel intensity 
%distribution. Fitting is performed using the psfFit_Image function from 
%TrackNTrace as described in:
%
%Stein, S., Thiart, J. TrackNTrace: A simple and extendable open-source 
%framework for developing single-molecule localization and tracking 
%algorithms. Sci Rep 6, 37947 (2016). https://doi.org/10.1038/srep37947
%
%
%[spotsFitted, nNonfittedSpots] = fit_spots(stack, spots, params,ui)
%
%Input:
% stack     -   3d-array of pixel values. 
%                   1st dimension (column): y-coordinate of the image plane 
%                   2nd dimension (row):    x-coordinate of the image plane 
%                   3rd dimension: frame number
%
% spots     -   Cell array that has as many cells as there are frames in 
%               the image stack. Each cell contains a 2-column array with
%               x-coordinates in the 1st column and y-coordinates in the
%               2nd column.
%               Eg.: spots{10} = [5.1,10.2; 20.5,30.1] implies that there
%               are two spots in the 10th frame of the stack, one at
%               (x=5.1, y=10.2) and one at(x=20.5, y=30.1).
%
% params     -  Struct with fields:
%                   minSpotDist:
%                       Minimum distance two spots have to be appart, if 
%                       closer together one of the two spots gets rejected
%                   maxRefinementDist:
%                       Maximum distance that is allow between the original
%                       and the refined position. If the distance is
%                       greater, the spot is fitted again within a smaller
%                       window (which helps eg. if two spots lie close
%                       together). If the distance after second fitting
%                       attempt is still larger than maxRefinementDist, the
%                       spot is discarded.
%                   gaussFitWidth:
%                       Defines the size of the window in which fitting is 
%                       performed. The side length of the window is 
%                       gaussFitWidth*2-1.                       
%                   maxWidth:
%                       Maximum variance of the fitten Gaussian function in
%                       x -and y-direction. If bigger than maxWidth, the 
%                       spot is discarted (use inf to use all spots).
%                   minWidth:
%                       Minimum variance of the fitten Gaussian function in
%                       x -and y-direction. If smaller than minxWidth, the 
%                       spot is discarted (use 0 to use all spots). 
%
%Output:
% nNonfittedSpots - Number of spots that were discarded because their
%                   refined position was farther away from the original
%                   position than maxRefinementDist, even after trying to
%                   fit it again in a smaller window.
% spotsFitted     - Cell array that has as many cells as there are frames in 
%                   the image stack. Each cell contains an array with 8
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
%
%
%OutputExample of spotsFitted: 
%   spotsFitted{10} = [5.1,10.2, 1044.3, 924.7, 1.48, -1, 0, 1;
%                     20.5, 30.1, 2358.9, 830.2, 0.76, -1, 0, 1]]
%   This mean that there are two spots in the 10th frame of the stack,
%   one at (x=5.1, y=10.2) and one at(x=20.5, y=30.1).
%   Maximum intensity, background and sigma_x are given in
%   columns 3,4 and 5. Columns 6 and 7 do not apply because we fit
%   an isotropic gaussian. Last row is the exitflag.
%                   




spotsFitted = spots;

minSpotDist = params.minSpotDist;
maxRefinementDist = params.maxRefinementDist;
gaussFitWidth = params.gaussFitWidth;
maxWidth = params.maxWidth;
minWidth = params.minWidth;

nFrames = size(stack,3);
%if nFrames>params.


nNonfittedSpots = 0;


%Define fit settings
param_optimizeMask = [1,1,1,1,1,0,0]; %[xpos,ypos,A,BG,sigma_x,sigma_y,angle]

%Iterate through all frames of the image stack
for i = 1:nFrames
    
    %Check if spots exist in current frame
    if ~isempty(spots{i})
        
        %--------1 Fit spots-------------------------------------------------       
        
        %Fit spots in current frame
        spotsFitted{i} = psfFit_Image(stack(:,:,i), spots{i}',param_optimizeMask,false,false,gaussFitWidth)';%[xpos,ypos,A,BG,sigma_x,sigma_y,angle; exitflag]
        %---------------------------------------------------------- 


        %--------2 Check if spots have jump too far after refinement---------
                
        % If new position is far away from local maximum (usually because
        % two spots are in close proximity and fit converges to the point between
        % the two spots). If yes, try again with smaller fitting window
        
        %Calculate jumping distance between original position and fitted position
        refinementDist    = sqrt((spots{i}(:,1) - spotsFitted{i}(:,1)).^2 + (spots{i}(:,2) - spotsFitted{i}(:,2)).^2);
        
        %Identify spots where refinement is larger than 
        spotsJump   = spots{i}(refinementDist > maxRefinementDist,:);
        jumpIdx     = refinementDist > maxRefinementDist;
        
        
        if ~isempty(spotsJump)       
            %Fit spots that jumped farther than maxRefinementDist
            results   =  psfFit_Image(stack(:,:,i), spotsJump',[1,1,1,1,1,0,0],false,false,1);
            
            %Write new fitted position
            spotsFitted{i}(jumpIdx,1:2) = results(1:2,:)';
            
            %Check again if jump is too large
            jumpDist2    = sqrt((spots{i}(:,1) - spotsFitted{i}(:,1)).^2 + (spots{i}(:,2) - spotsFitted{i}(:,2)).^2);
            spotsJump2   = spots{i}(jumpDist2 > maxRefinementDist,:);
            jumpIdx2     = jumpDist2 > maxRefinementDist;
            
            %Discard spots where refined position is still further away
            %from original position than maxRefinementDist
            if ~isempty(spotsJump2)
                spotsFitted{i}(jumpIdx2,:) = [];
            end
            
            nNonfittedSpots = nNonfittedSpots + 1;
        end
        %-------------------------------------------------------------
                
        %Enable to show only spots where fit jump was too high
%         if ~isempty(spotsJump)
%             spotsFitted{i}(~jumpIdx,:) = [];
%         else
%             spotsFitted{i} = zeros(0,2);
%         end

        %--------3 Check width of the fitted gaussians-----------------------    
        % 这一功能有没用上
        %Discard spots which are smaller/bigger than params.minWidth/params.maxWidth (currently not used)
        if maxWidth ~= inf
            spotsFitted{i} = spotsFitted{i}(spotsFitted{i}(:,5)<maxWidth,:);
            if spotsFitted{i}(1,6) ~= -1
                spotsFitted{i} = spotsFitted{i}(spotsFitted{i}(:,6)<maxWidth,:);
            end
        end
        
        if minWidth ~= 0
            spotsFitted{i} = spotsFitted{i}(spotsFitted{i}(:,5)>minWidth,:);
            if spotsFitted{i}(1,6) ~= -1
                spotsFitted{i} = spotsFitted{i}(spotsFitted{i}(:,6)>minWidth,:);
            end
        end
        %-------------------------------------------------------------

    end
    
    %--------4 Check if spots are too close together-------------------------
    % 这部分就可以可视化出来，只选上亮度最大的那一个点
    if ~isempty(spotsFitted{i})
        if i == 488
            size(spotsFitted{i}, 1);
        end
%         if i == 489
%             size(spotsFitted{i}, 1);
%         end
%         if i == 490
%             size(spotsFitted{i}, 1);
%         end
%         if i == 491
%             size(spotsFitted{i}, 1);
%         end

        %Calculate x and y distances between all spots in current frame
        xDists = repmat(spotsFitted{i}(:, 1), 1, size(spotsFitted{i}, 1)) ...
            - repmat(spotsFitted{i}(:, 1)', size(spotsFitted{i}, 1), 1);
        yDists = repmat(spotsFitted{i}(:, 2), 1, size(spotsFitted{i}, 1)) ...
            - repmat(spotsFitted{i}(:, 2)', size(spotsFitted{i}, 1), 1);
        
        %Caclulate distances between all spots
        dists = triu(sqrt(xDists.^2 + yDists.^2));
        
        %Find spots which are closer together than params.minSpotDist
        dists(dists > minSpotDist) = 0;
        
        %Get intensities of nearby spots and discard spot with lower intensity
        if any(dists(:))
            [idxSpot1, idxSpot2] = find(dists);
            intSpot1 = stack(round(spotsFitted{i}(idxSpot1,2)),round(spotsFitted{i}(idxSpot1,1)),i);
            intSpot2 = stack(round(spotsFitted{i}(idxSpot2,2)),round(spotsFitted{i}(idxSpot2,1)),i);
            
            if intSpot1 > intSpot2
                spotsFitted{i}(idxSpot2,:) = [];
            else
                spotsFitted{i}(idxSpot1,:) = [];
            end
            
        end
    end
    
    %------Monitor progress------------------------------------------------
    

end



end








