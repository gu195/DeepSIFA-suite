function tiffMatrix = tiffLoadStack(fileName)
% 读取3维tiff视频
%Input:
%  pathName     -   Character array defining the path where the tiff file
%                   is located.
%  fileName     -   Character array defining the name of the tiff file

%Output:
%   tiffMatrix -   3d-array of pixel values. 
%                       1st dimension (column): y-coordinate of the image plane 
%                      	2nd dimension (row):    x-coordinate of the image plane 
%                      	3rd dimension: frame number





%Full directory of .tiff file
fullPath = fileName;
if exist(fullPath, 'file') == 2 %Check if file exists
    %Supress warnings for unrecognized tif tags
    warning('off'); 
    
    %Set empty current character to check for escape-button hit during
    %loading procedure
    %set(gcf,'currentch',char(1)); %
%Create link to desired tiff file
    TifLink = Tiff(fullPath, 'r');
    
    %Get image size
    mImage       = TifLink.getTag('ImageWidth');
    nImage       = TifLink.getTag('ImageLength');
    
    %Get data type of pixel values
    switch TifLink.getTag('SampleFormat')
        case Tiff.SampleFormat.UInt
            switch TifLink.getTag('BitsPerSample')
                case 8
                    bitDepth = 'uint8';
                case 16
                    bitDepth = 'uint16';
                case 32
                    bitDepth = 'uint32';
            end
        case Tiff.SampleFormat.Int
            switch TifLink.getTag('BitsPerSample')
                case 8
                    bitDepth = 'int8';
                case 16
                    bitDepth = 'int16';
                case 32
                    bitDepth = 'int32';
            end
        case Tiff.SampleFormat.IEEEFP
            switch TifLink.getTag('BitsPerSample')
                case 32
                    bitDepth = 'single';
                case 64
                    bitDepth = 'double';
            end
    end
    
    %Initialize the tiff stack with a fixed number of frames
    %as we don't know the total number of frames yet
    initSize = 5000;
    stepSize = 10000;
    tiffMatrix=zeros(nImage,mImage,initSize,bitDepth);
    
    %Initialize the maximum amount of frames of tiffMatrix, which will be
    %increased continually 
    imax = stepSize;
    
    i = 1;
    while true
        %Check if current frame number is higher than the tiffMatrix size
        if i > imax
            %Increase size of our tiffMatrix by the stepSize
            imax = imax + stepSize;
            tiffMatrix(:,:,imax) = 0;
        end
        
        %Save current image plane to tiffMatrix
        tiffMatrix(:,:,i)=TifLink.read();
        
        %Every 100th frame: update loading progress in the user interface
        %and check if user pressed Escape
        if mod(i,100) == 0
            msg = sprintf('Loading stack: %3.0f frames loaded', i);
            %ui.editFeedbackWin.String = char(msg, origFeedbackWin);
            drawnow
        end
        
        %if double(get(gcf,'CurrentCharacter')) == 24
        %    % User pressed strg+s so stop loop
        %    tiffMatrix = [];            
        %    break
        %end
                
        if TifLink.lastDirectory()
            %We reached the last image plane
            %We probably initialized more frames than we needed so use only frames
            %that have been defined in loop and discard the rest
            tiffMatrix = tiffMatrix(:,:,1:i);
            %ui.editFeedbackWin.String = origFeedbackWin;
            break
        else
            %Go to next image plane
            TifLink.nextDirectory();
            i = i + 1;
        end
    end
    
    
    %Close link to tiff file
    TifLink.close();
    %ui.editFeedbackWin.String = origFeedbackWin;
    %Re-enable warnings
    warning('on');
else
    %File was not found so return an emtpy tiffMatrix
    tiffMatrix = [];
    %ui.editFeedbackWin.String = 'Movie not found';
end


end
