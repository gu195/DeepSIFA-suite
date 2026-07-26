function [spots,stdWaveletAll,threshold, filtered] = spotsFind(stack, thresholdFactor, frameRange, ROI)
% 在单分子图像堆栈中通过小波滤波和图像扩张（dilation）技术检测光点（spots）。
% 光点是指那些局部最大值的像素点，并且这些像素点的值需要高于特定的阈值。
% 也涉及到对ROI内的图像数据进行筛选和处理，以提高效率和精确度。
%
% Input:
%   stack           -   3d-array of pixel values. 
%                           1st dimension (column): y-coordinate of the image plane 
%                           2nd dimension (row):    x-coordinate of the image plane 
%                           3rd dimension: frame number
%   thresholdFactor:    used to calculate the detection threshold.
%   frameRange:         array containing first and last frame
%                       of the image stack that should be
%                      	analysed
%   ROI             -   2-column vector containing coordinates of a region of interest where                    
%                       the spots should be detected. 1st column x-coordinates (horizontal),
%                       2nd column y-coordinates (vertical).
%
% Output:
%   spots           -   Cell array that has as many cells as there are
%                       frames in the image stack. Each cell contains a
%                       2-column array with x-coordinates in the 1st column
%                       and y-coordinates in the 2nd column.
%                       Eg. spots{10} = [5.1,10.2; 20.5,30.1] implies that
%                       there are two spots in the 10th frame of the stack,
%                        one at (x=5.1, y=10.2) and one at(x=20.5, y=30.1). 
%   stdWaveletAll   -   Estimated background noise which was used to
%                       calculate the detection threshold
%   threshold       -   Threshold above which spots are detected



%图片尺寸x，y，n
stackSize = size(stack);

%frames
nFrames = stackSize(3);

%确保要分析的图片范围不超过总帧数nframes
if frameRange(2) > nFrames
    frameRange(2) = nFrames;
end

%暂存ROI xy坐标
ROIx = ROI(:,1);
ROIy = ROI(:,2);

%----Cut out ROI of original stack to increase speed-----------------------

%Increase ROI to reduce border effects when filtering
additionalPixels = 5;

roiMaxX = ceil(min(max(ROIx)+additionalPixels,stackSize(2)));
roiMinX = floor(max(min(ROIx)-additionalPixels,1));
roiMaxY = ceil(min(max(ROIy)+additionalPixels,stackSize(1)));
roiMinY = floor(max(min(ROIy)-additionalPixels,1));

%读取整个图片
%if min(ROIx) == 0 || min(ROIy) ==0
%    roiMaxX = stackSize(2);
%    roiMinX = 1;
%    roiMaxY = stackSize(1);
%    roiMinY = 1;

if ~isa(stack,'double') && ~isa(stack,'single')
    filtered = single(stack(roiMinY:roiMaxY,roiMinX:roiMaxX,:));
else
    filtered = stack(roiMinY:roiMaxY,roiMinX:roiMaxX,:);
end

filtered_uint16 = stack;
binary_1_uint16 = stack;
binary_2_uint16 = stack;
binary_3_uint16 = stack;
%--------------------------------------------------------------------------

%Initialize cell array for spots
spots = repmat({zeros(0,2)},nFrames,1);

%Initialize variable to save the framewise standard deviations of the first
%wavelet filtered images
stdWavelet1 = zeros(nFrames,1);

drawnow


% Filter stack with wavelet filter,这段代码先注释起来，现在要可视化中间过程，attention
% % 小波滤波器里面的标准差可以协助阈值调整
parfor k = frameRange(1):frameRange(2)
    [filtered(:,:,k), stdWavelet1(k)] = wavelet_filter(filtered(:,:,k));
end


% % ---------------------------可视化  保存小波滤波后的结果0210------------------------------
% for k = frameRange(1):frameRange(2)
%     [filtered(:,:,k), stdWavelet1(k)] = wavelet_filter(filtered(:,:,k));
% end
% 
% for k = frameRange(1):frameRange(2)
%     % 将 filtered 转换为 uint16 类型
%     filtered_uint16(:,:,k) = uint16(filtered(:,:,k));
% end
% 
% outputPath = 'filtered_image_stack0210_4.tif';  % 输出文件路径
% % 使用 'WriteMode' 来保存每一帧到同一个文件中
% for k = 1:size(filtered_uint16, 3)
%     if k == 1
%         % 对于第一帧，创建新文件
%         imwrite(filtered_uint16(:,:,k), outputPath, 'WriteMode', 'overwrite', 'Compression', 'none');
%     else
%         % 对于后续帧，附加到同一文件
%         imwrite(filtered_uint16(:,:,k), outputPath, 'WriteMode', 'append', 'Compression', 'none');
%     end
% end
% % ----------------------------------保存小波滤波后的结果-----------------------------------


%Stdev of the first order wavelet map. Used to estimate background noise.
stdWaveletAll = round(mean(stdWavelet1(stdWavelet1 ~= 0)),2);

%Calculate threshold for spot finding based on the estimated background
%noise of the first order wavelet image.
threshold = stdWaveletAll * 0.01 * thresholdFactor;% attention 0124
% threshold = thresholdFactor;


% %     % ----------------------------------可视化 保存检测候选对象的结果0211----------------------------------- 
% for k = frameRange(1):frameRange(2)
%     curFrameFiltered = filtered(:,:,k);    
%     %----------Local maxima finding through image dilation-----------------
%     %Image dilation mask
%     se = strel('square',4);
%     if k == 490
%         se = strel('square',4);
%     end
% 
%     % 对当前帧进行膨胀操作
%     dilated_frame = imdilate(curFrameFiltered, se);
%     binary_1_uint16(:,:,k) = uint16(dilated_frame);
% 
%     % 比较膨胀后的图像和原图，找出局部最大值
%     % 在该二值图像中，原图和膨胀图像相等的位置（即局部最大值的位置）会被标记为1，其余位置为0。
%     binary = curFrameFiltered == dilated_frame;
%     binary_2_uint16(:,:,k) = uint16(binary);
%     
%     %Set spots below the user defined threshold to zero
%     binary(curFrameFiltered < threshold) = 0;
%     binary_3_uint16(:,:,k) = uint16(binary);
% 
%     %Find spot positions
%     [spotsY, spotsX] = find(binary);
%     
%     %Calculate position on non-cropped image
%     spotsX = spotsX + roiMinX-1;
%     spotsY = spotsY + roiMinY-1;
%         
%     %Look up which spots are inside ROI
%     inROI = inpolygon(spotsX,spotsY,ROIx,ROIy);
%     %inROI = 1;
%     spotsCurFrame = [spotsX spotsY];
%     
%     %Save only spots which are inside ROI
%     spots{k} = spotsCurFrame(inROI == 1,:);
% end
%     % ----------------------------------保存检测候选对象的结果----------------------------------- 


parfor k = frameRange(1):frameRange(2)
    curFrameFiltered = filtered(:,:,k);
    
    %----------Local maxima finding through image dilation-----------------
    %Image dilation mask
    se = strel('square',4);
    
%     Identify local maxima as pixels which have the same value before and after image dilation
%     看一下imdilate的逻辑，attention
    binary = curFrameFiltered == imdilate(curFrameFiltered,se);
    binary(curFrameFiltered < threshold) = 0;
    
    %Find spot positions
    [spotsY, spotsX] = find(binary);
    
    %Calculate position on non-cropped image
    spotsX = spotsX + roiMinX-1;
    spotsY = spotsY + roiMinY-1;
        
    %Look up which spots are inside ROI
    inROI = inpolygon(spotsX,spotsY,ROIx,ROIy);
    %inROI = 1;
    spotsCurFrame = [spotsX spotsY];
    
    %Save only spots which are inside ROI
    spots{k} = spotsCurFrame(inROI == 1,:);
end


% % ----------------------------可视化  保存检测候选对象的结果0211---------------------------- 
% outputPath = 'binary_image_stack0211_1.tif';  % 输出文件路径
% % 使用 'WriteMode' 来保存每一帧到同一个文件中
% for k = 1:size(binary_1_uint16, 3)
%     if k == 1
%         % 对于第一帧，创建新文件
%         imwrite(binary_1_uint16(:,:,k), outputPath, 'WriteMode', 'overwrite', 'Compression', 'none');
%     else
%         % 对于后续帧，附加到同一文件
%         imwrite(binary_1_uint16(:,:,k), outputPath, 'WriteMode', 'append', 'Compression', 'none');
%     end
% end
% 
% outputPath = 'binary_image_stack0211_2.tif';  % 输出文件路径
% % 使用 'WriteMode' 来保存每一帧到同一个文件中
% for k = 1:size(binary_2_uint16, 3)
%     if k == 1
%         % 对于第一帧，创建新文件
%         imwrite(binary_2_uint16(:,:,k), outputPath, 'WriteMode', 'overwrite', 'Compression', 'none');
%     else
%         % 对于后续帧，附加到同一文件
%         imwrite(binary_2_uint16(:,:,k), outputPath, 'WriteMode', 'append', 'Compression', 'none');
%     end
% end
% 
% outputPath = 'binary_image_stack0211_3.tif';  % 输出文件路径
% % 使用 'WriteMode' 来保存每一帧到同一个文件中
% for k = 1:size(binary_3_uint16, 3)
%     if k == 1
%         % 对于第一帧，创建新文件
%         imwrite(binary_3_uint16(:,:,k), outputPath, 'WriteMode', 'overwrite', 'Compression', 'none');
%     else
%         % 对于后续帧，附加到同一文件
%         imwrite(binary_3_uint16(:,:,k), outputPath, 'WriteMode', 'append', 'Compression', 'none');
%     end
% end
% % ----------------------------------保存检测候选对象的结果-----------------------------------


end






