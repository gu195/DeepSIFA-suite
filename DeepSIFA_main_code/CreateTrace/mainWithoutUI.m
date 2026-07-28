function result = mainWithoutUI(filename, workdir, parameter)

% fileName = "H:\Img_Proc\motion_img\film1.tif";

result = 0;
fullpath = filename;
imgStack = tiffLoadStack(fullpath);
[batchInit,~] = init_batch();
nFiles = 1;
newBatch = repmat(batchInit, nFiles, 1);

batch = newBatch;
%% 以下定义参数
para = struct();
para.frameRange = [parameter.nStart parameter.nEnd];% 要分析的帧数范围，inf表示所有帧

para.thresholdFactor = parameter.threshold;         % 阈值默认85
para.trackingRadius = parameter.spotTrackingRadius; % 亮点最大跳跃距离，默认3px
para.gaussFitWidth = parameter.gaussFitWidth;       % 高斯拟合宽度控制，2*gaussFitwidth-1，默认3pX attention，视乎没有传入trackingSM的spotsFit(stack, spots, batch(n).params);

para.minTrackLength = parameter.frameLength;        % 光斑最小持续的帧数，默认20
para.gapFrames = parameter.frameGap;                % 描述光斑可能不连续的最大帧数，默认θ

% 分析移动斑点的方法
para.trackingMethod = parameter.trackMethod;
if strcmp(para.trackingMethod, 'u-track')
    addpath(genpath('u-track\software'));
    para.utrackMotionType = parameter.utrackMotionType;
end




batch(1).movieInfo.fileName = fullpath;

[batch,filtered] = trackingSM(batch, para, imgStack);

getResultsFromBatch(batch, workdir,filtered);

result = 1;

function getResultsFromBatch(batch, pathname,filtered)
    %% 结果文件, 保存到图片同目录中data文件夹下, 
    % 每个文件表示1个移动的亮点，共11列
    % 1: 帧数n, 2-3：x坐标(px), y坐标(px), 
    % 4-5: Intensity, Background Intensity,
    % 6-7: (高斯平滑中标准差)sigma_x, sigma_y(=-1, isotropic for y direction) 
    % 8: 平滑方向，无意义！！
    % 9: flags, 1 means no error, -1 means failed
    % 10: jump angles (degree)
    % 11: jump distance (px)
    
    % 组合成完整的 data 文件夹路径
    dataFolderPath = fullfile(pathname, 'data');

    % 检查 data 文件夹是否存在
    if ~exist(dataFolderPath, 'dir')
    % 如果不存在，创建 data 文件夹
        mkdir(dataFolderPath);
        disp(['Begin--Created folder: ', dataFolderPath]);
    else
        disp(['Folder already exists: ', dataFolderPath]);
    end

    %从batch中读取results，并转化为2维矩阵形式
    tracksAll = batch.results.tracks;
    angles = batch.results.angles;
    jumpDist = batch.results.jumpDistances;
    startEndFrameOfTracks = batch.results.startEndFrameOfTracks;

    % 遍历每个 cell，将其写入一个新的 sheet
    for i = 1:numel(tracksAll)
        formatSpec = '正在写入第 %d个轨迹，共%d 个\n';
        processOut = sprintf(formatSpec,i, numel(tracksAll));
        % fprintf(processOut);

        % 获取tracksAll的2D数据
        data = tracksAll{i};
        % 获取 angles和jumpDist中对应的列数据
        colData1 = angles{i};
        colData2 = jumpDist{i};
    
        % 确定当前 sheet 的最大行数
        maxRows = max(size(data, 1), size(colData1, 1));
    
        % 创建一个新的矩阵来合并 data 和 colData1-2
        combinedData = NaN(maxRows, size(data, 2) + 2);
    
        % 将 data 插入到 combinedData 的左侧
        combinedData(1:size(data, 1), 1:size(data, 2)) = data;
    
        % 将 colData1 插入到 combinedData 的右侧
        combinedData(1:size(colData1, 1), end-1) = colData1;
        combinedData(1:size(colData2, 1), end) = colData2;

%         % --------------------补充第一列不连续的数字 0212感觉没有道理---------------------
%         new_combinedData = [];  % 用于存储新矩阵
%         for i_ = 1:size(combinedData, 1) - 1
%             % 将当前行添加到新矩阵
%             new_combinedData = [new_combinedData; combinedData(i_, :)];
%             
%             % 检查当前行与下一行的第一列数字差异
%             current_value = combinedData(i_, 1);
%             next_value = combinedData(i_+1, 1);
%             
%             if next_value > current_value + 1
%                 % 如果数字不连续，则插入缺失的行
%                 for j_ = current_value + 1 : next_value - 1
%                     % 创建一行，复制当前行的内容
%                     new_row = combinedData(i_, :);
%                     new_row(1) = j_;  % 只更新第一列数字
%                     new_combinedData = [new_combinedData; new_row];  % 将新行添加到新矩阵
%                 end
%             end
%         end
%         % 添加最后一行
%         new_combinedData = [new_combinedData; combinedData(end, :)];
%         
%         % 更新 combinedData
%         combinedData = new_combinedData;
%         % ------------------------------------------------------
        



        %% 以上为第一个sheet的数据
        %% 以下为sheet2数据, 包括 帧数n x坐标(px) y坐标(px) gaussIntegralIntensity  sumSqureIntensity  sumCircleIntensity
        
        x0 = combinedData(:,2); % 圆心/中心 x 坐标序列
        y0 = combinedData(:,3); % 圆心/中心 y 坐标序列
        Amp = combinedData(:,4); BG = combinedData(:,5); 
        sigmax = combinedData(:,6); 
        sigmax(isnan(sigmax)) = 1; BG(isnan(BG)) = 800;
        
        % 高斯拟合函数
        gauss2DFit = @(x,y,x0,y0,Amp,BG,sigmax)  Amp.*exp(-(x-x0) .^ 2 ./ (2 .*sigmax.^2)-(y-y0).^2 ./ 2) + BG;

        r = double(para.gaussFitWidth) * 0.5; % 半径/方形半宽

        index = combinedData(:,1); %获取当前帧数
        curFrameInt = imgStack(:,:,index);

        % 通过2维高斯拟合函数积分计算圆形区域总强度
        gaussIntegralIntensity = integrate_Gauss2D_in_circles(gauss2DFit, y0, x0, Amp, BG, sigmax, r);


        % 能不能integrate_Gauss2D_in_circles追踪全过程
        imgLength = size(imgStack,3);
        xCoord = zeros(1,imgLength); yCoord = xCoord; Amp=xCoord; sigmax = xCoord; 
        xCoord(1:index(end)) = x0(1); xCoord(index(end):end)=x0(end);
        yCoord(1:index(end)) = y0(1); yCoord(index(end):end)=y0(end);         
        xCoord(index)=x0;yCoord(index)=y0;
        %index的部分为x0 y0 缺少点选择最接近的点会不会更加准确一点

%         % ------------------------------间隙帧直接不要---------------------------------------
%         imgLength = size(imgStack,3);
%         xCoord = zeros(1,imgLength); yCoord = xCoord; Amp=xCoord; sigmax = xCoord;        
%         xCoord(index)=x0;yCoord(index)=y0;
%         % --------------------------------------------------------------------------------

        sumSqureIntensity = zeros(1,imgLength);
        % 计算n*n方形区域的总强度，直接相加!!!
        sumSqureIntensity = sum_weighted_square_region(imgStack, yCoord', xCoord', r*2);
        % gaussIntegralIntensity_all = integrate_Gauss2D_in_circles(gauss2DFit, yCoord', xCoord', Amp, BG, sigmax, r);attention       



%         %计算圆形区域总强度，直接相加!!!
%         sumCircleIntensity_all = circular_intensity(imgStack, yCoord', xCoord', r+2);
%         sumCircleIntensity_sub = circular_intensity(imgStack, yCoord', xCoord', r);
%         sumCircleIntensity_background = sumCircleIntensity_all - sumCircleIntensity_sub;
%         sumCircleIntensity = sumCircleIntensity_sub -sumCircleIntensity_background;

%         %计算圆形区域总强度，直接相加!!!
%         filtered_uint16 = uint16(filtered);
%         sumCircleIntensity_all = circular_intensity(filtered_uint16, yCoord', xCoord', r+2);
%         sumCircleIntensity_sub = circular_intensity(filtered_uint16, yCoord', xCoord', r);
%         sumCircleIntensity_background = sumCircleIntensity_all - sumCircleIntensity_sub;
%         sumCircleIntensity = sumCircleIntensity_sub -sumCircleIntensity_background;

        %计算圆形区域总强度，直接相加!!!
        % 1
        % filtered_uint16 = uint16(filtered);
        % sumCircleIntensity = circular_intensity(filtered_uint16, yCoord', xCoord', r, r+2);%x
        % 2
        % sumCircleIntensity = circular_intensity(imgStack, yCoord', xCoord', r, r+2);
        % 3
        imgGauss = imgaussfilt(imgStack, 2);
        sumCircleIntensity = circular_intensity(imgGauss, yCoord', xCoord', r, r+2);
%         % 4
%         imgGauss = imgaussfilt(imgStack, 1);
%         xCoord_uint = round(xCoord);  % 四舍五入 xCoord 为整数
%         yCoord_uint = round(yCoord);  % 四舍五入 yCoord 为整数
%         sumCircleIntensity = circular_intensity(imgGauss, yCoord_uint', xCoord_uint', r, r+2);


%         %计算圆形区域总强度，直接相加!!!
%         sumCircleIntensity_all = circular_intensity(imgStack, yCoord', xCoord', r+2);
%         sumCircleIntensity_sub = circular_intensity(imgStack, yCoord', xCoord', r);
%         sumCircleIntensity_background = sumCircleIntensity_all - sumCircleIntensity_sub;
%         sumCircleIntensity_sub(index(:)) =gaussIntegralIntensity(:);
%         sumCircleIntensity = sumCircleIntensity_sub -sumCircleIntensity_background;
        
    
        % 生成 sheet 名称，例如 'Sheet1', 'Sheet2', ..., 'Sheet10'
        %sheetName = ['Sheet', num2str(i)];
        % 生成文件名，例如 'track1.csv', 'data100.csv'
        filename1 = ['/data/track', num2str(i), '.csv'];
        fullpath = fullfile(pathname, filename1);    
        % 使用 writematrix 将数据写入 Excel 文件中, 仅包含tracks
        writematrix(round(combinedData, 2), fullpath); 

        filename2 = ['/data/allFramesTrackInten', num2str(i), '.csv'];
        fullpath2 = fullfile(pathname, filename2);
        % attention 可视化不同的强度计算方法
        % final_gauss = sumSqureIntensity;% attention
        final_gauss = sumCircleIntensity;% attention
        %final_gauss(index(:)) =gaussIntegralIntensity(:);%!高斯平滑信噪比会好点儿，看一下怎么加入合理 attention


        x0_length = length(x0); % 获取 x0 的长度
        y0_length = length(y0); % 获取 x0 的长度


        % 根据条件设置开始位置
        if index(1) > 1000
            startIdx = index(1) - 500; 
        elseif index(1) > 800 && index(1) <= 1000
            startIdx = index(1) - 400;
        elseif index(1) > 600 && index(1) <= 800
            startIdx = index(1) - 300; 
        elseif index(1) > 400 && index(1) <= 600
            startIdx = index(1) - 250; 
        elseif index(1) > 200 && index(1) <= 400
            startIdx = index(1) - 200; 
        elseif index(1) > 150 && index(1) <= 200
            startIdx = index(1) - 90; 
        elseif index(1) > 100 && index(1) <= 150
            startIdx = index(1) - 70; 
        else
            startIdx = 1; 
        end

        
        % 根据 index(end) 的范围设置 endIdx
        index_wave = index(end)-index(1);  
        if index_wave <= 30
            endIdx = min(index(end) + 260, imgLength); 
        elseif index_wave > 30 &&  index_wave <= 40 
            endIdx = min(index(end) + 255, imgLength); 
        elseif index_wave > 40 &&  index_wave <= 50
            endIdx = min(index(end) + 250, imgLength); 
        elseif index_wave > 50 &&  index_wave <= 70
            endIdx = min(index(end) + 245, imgLength); 
        elseif index_wave > 70 &&  index_wave <= 100
            endIdx = min(index(end) + 240, imgLength); 
        elseif index_wave > 100 &&  index_wave <= 120
            endIdx = min(index(end) + 230, imgLength); 
        elseif index_wave > 120 &&  index_wave <= 150
            endIdx = min(index(end) + 225, imgLength); 
        else
            endIdx = min(index(end) + 200, imgLength);
        end


        %----------0213 强度前后延伸------------------
        xCoord_ = xCoord(startIdx:endIdx);
        yCoord_ = yCoord(startIdx:endIdx);
        sumSqureIntensity_ = sumSqureIntensity(startIdx:endIdx); 
        sumCircleIntensity_ = sumCircleIntensity(startIdx:endIdx); 
        final_gauss_ = final_gauss(startIdx:endIdx); 
        bg_mean = mean(BG(:)); % 转换为列向量计算整体平均值
        bg_mean = repmat(bg_mean, endIdx-startIdx+1, 1); % 复制 bg_mean 为 imgLength 行的列向量
        intensityData = [...
            (startIdx:endIdx)', ...
            round(xCoord_', 2), ...
            round(yCoord_', 2), ...
            round(bg_mean, 2), ...
            round(sumSqureIntensity_', 2), ...
            round(sumCircleIntensity_', 2), ...
            round(final_gauss_', 2) ...
        ];
%         writematrix(intensityData, fullpath2); 
        
%         selectedRows = intensityData(index, :); % 只选择 index 对应的行  
%         writematrix(selectedRows, fullpath2);   % 将筛选后的数据保存
%         %应该是startIdx到index的第一行，index最后一行到endIdx这些行都要显示出来 
        % 看图片的效果感觉好像开了荧光团连接
        if i == 120
            postIndexData = [];
        end
        distance = index(1)-startIdx;
        if index(1) > 2
            preIndexData = intensityData(1:distance, :);% 提取 startIdx 到 index(1) 的第一行的数据 
        else
            preIndexData = [];
        end    
        indexData = intensityData(index-startIdx+1, :);% 提取 index 对应的行的数据

        % 确保索引不超过数据总长度
        if index(end) <= size(intensityData, 1)
            postIndexData = intensityData(index(end)-startIdx+2:endIdx-startIdx+1, :);
        else
            postIndexData = [];
        end 

        combinedData = [preIndexData; indexData; postIndexData];        
        writematrix(combinedData, fullpath2);
        %----------0213 强度前后延伸------------------
    end


  end

  function integral_values = integrate_Gauss2D_in_circles(f, x0, y0, Amp, BG, sigmax, r)
    % f: 函数句柄，表示 f(x, y, x0)
    % x0, y0: 圆心坐标的序列，长度均为 100
    % r: 半径
    
    % 确保 x0 和 y0 是相同长度的向量
    assert(length(x0) == length(y0) && length(y0) == length(Amp) && ...
           length(Amp) == length(BG) && length(BG) == length(sigmax), ...
           '所有参数序列必须具有相同长度');
    
    % 初始化存储积分结果的向量
    num_points = length(x0);
    integral_values = zeros(num_points, 1);
    
    % 定义极坐标下的积分函数
     integrand = @(theta, rho, x0i, y0i, Ampi, BGi, sigmaxi) ...
        f(x0i + rho .* cos(theta), y0i + rho .* sin(theta), ...
        x0i, y0i, Ampi, BGi, sigmaxi) .* rho;
    
    % 设置积分范围
    theta_min = 0;
    theta_max = 2 * pi;
    rho_min = 0;
    rho_max = r;
    
    % 对每个圆心进行积分
    for i = 1:num_points
        x0i = x0(i);
        y0i = y0(i);
        Ampi = Amp(i);
        BGi = BG(i);
        sigmaxi = sigmax(i);
        
        % 定义当前圆心的积分函数
        current_integrand = @(theta, rho) integrand(theta, rho, x0i, y0i, Ampi, BGi, sigmaxi);
        
        % 计算积分
        integral_values(i) = round(integral2(current_integrand, theta_min, theta_max, rho_min, rho_max));
    end
  end

      function result = sum_square_region(matrix, x0, y0, n)
        % matrix: 输入的矩阵
        % x0, y0: 浮点数坐标序列
        % n: 方形区域的边长（正的有理数）
    
        % 输入参数检查
        assert(mod(n, 2) == 1, 'n 必须是奇数');
        assert(length(x0) == length(y0), 'x0 和 y0 必须长度相同');
    
        % 初始化结果数组
        num_points = length(x0);
        result = zeros(1, num_points);
    
        % 方形区域的半边长
        half_n = (n - 1) / 2;
    
        % 对每对 (x0[i], y0[i]) 进行处理
        for k = 1:num_points
            x_center = x0(k);
            y_center = y0(k);
            % 当前帧
            frame = k;
    
            % 确定区域的边界
            x_min = max(floor(x_center) - half_n, 1);
            x_max = min(floor(x_center) + half_n, size(matrix(:,:,k), 1));
            y_min = max(floor(y_center) - half_n, 1);
            y_max = min(floor(y_center) + half_n, size(matrix(:,:,k), 2));
    
            % 初始化单个方形区域的总和
            total_sum = 0;
    
            % 遍历选定区域内的每个元素，计算总和
            for i = x_min:x_max
                for j = y_min:y_max
                    total_sum = total_sum + matrix(i, j, frame);
                end
            end
    
            % 存储在结果数组中
            result(k) = total_sum;
        end
    end

    function result = sum_weighted_square_region(matrix, x0, y0, n)
        % matrix: 输入的矩阵
        % x0, y0: 浮点数坐标
        % n: 区域大小（正整数）
        
        % 输入参数检查
        assert(n > 0, 'n 必须是正整数');
    
        % 计算区域的半径 attention
        radius = (n - 1) / 2;            
                
        % 初始化结果数组
        num_points = length(x0);
        result = zeros(1, num_points);
        
        % 遍历选定区域内的每个元素，计算权重并加权求和
        for k = 1:num_points
            % 初始化权重矩阵和加权和
            weighted_sum = 0;
            total_weight = 0;
            x_center = x0(k);
            y_center = y0(k);
            % 确定区域的边界
            x_min = max(floor(x_center - radius), 1);
            x_max = min(ceil(x_center + radius), size(matrix, 1));
            y_min = max(floor(y_center - radius), 1);
            y_max = min(ceil(y_center + radius), size(matrix, 2));
            % 当前帧
            frame = k;
            for i = x_min:x_max
                for j = y_min:y_max
                % 计算每个矩阵元素的边界
                x1 = max(i - 0.5, x_center - radius);
                x2 = min(i + 0.5, x_center + radius);
                y1 = max(j - 0.5, y_center - radius);
                y2 = min(j + 0.5, y_center + radius);
                
                % 计算覆盖面积（权重）
                area = (x2 - x1) * (y2 - y1);
                
                % 累加加权和
                weighted_sum = weighted_sum + matrix(i, j, frame) * area;
                total_weight = total_weight + area;
                end
            end
            result(k) = weighted_sum;
        end
       
    end


function result = circular_intensity(matrix, x0, y0, r1, r2)
    % matrix: 输入的矩阵 (图像)
    % x0, y0: 圆心坐标序列 (浮动坐标)
    % r: 半径（正的有理数）
    % 黄色圈r1里边的强度全部加起来，减去背景平均值(圆环区域 r1<坐标<=r2)乘以黄色圈内的像素个数

    % 检查输入的参数有效性
    assert(r1 > 0, 'r 必须是正的有理数');
    assert(length(x0) == length(y0), 'x0 和 y0 必须长度相同');

    % 初始化结果数组
    num_points = length(x0);
    result = zeros(1, num_points);

    % 初始化 mask
    mask_ring_matrix = zeros(size(matrix, 1), size(matrix, 2), num_points);  % 用于存储圆环区域的mask
    mask_circle_matrix = zeros(size(matrix, 1), size(matrix, 2), num_points);  % 用于存储圆形区域的mask

    % 遍历每对圆心 (x0[i], y0[i])
    for k = 1:num_points
        if k == 424
            x_center = x0(k);
        end
        x_center = x0(k);
        y_center = y0(k);

        % 确定圆形区域的边界
        r1_x_min = max(floor(x_center - r1), 1);
        r1_x_max = min(ceil(x_center + r1), size(matrix, 1));
        r1_y_min = max(floor(y_center - r1), 1);
        r1_y_max = min(ceil(y_center + r1), size(matrix, 2));

        r2_x_min = max(floor(x_center - r2), 1);
        r2_x_max = min(ceil(x_center + r2), size(matrix, 1));
        r2_y_min = max(floor(y_center - r2), 1);
        r2_y_max = min(ceil(y_center + r2), size(matrix, 2));

        % 初始化该圆形区域(r=3)的总强度
        r1_intensity = 0;
        r1_num_pixels = 0;
        % 初始化圆环区域(3<r<=5)结果
        ring_intensity = 0;
        ring_num_pixels = 0; 

        % 生成 mask 用于标识哪些区域计算了像素
        mask_ring = zeros(size(matrix, 1), size(matrix, 2));
        mask_circle = zeros(size(matrix, 1), size(matrix, 2));

        % 计算圆环区域的强度
        for i = r2_x_min:r2_x_max
            for j = r2_y_min:r2_y_max
                distance = sqrt((x_center - i)^2 + (y_center - j)^2);    
                % 如果距离在 r1 和 r2 之间，则是圆环区域
                if distance >= r1 && distance <= r2
                    % 累加像素值和像素个数
                    ring_intensity = ring_intensity + matrix(i, j, k);
                    ring_num_pixels = ring_num_pixels + 1;
                    mask_ring(i, j) = 1;  % 该区域在圆环内，标记为1
                end
            end
        end
        % 计算背景强度（圆环区域的平均强度）
        BG = ring_intensity / ring_num_pixels; % 计算背景强度

        % 计算圆形区域的强度
        for i = r1_x_min:r1_x_max
            for j = r1_y_min:r1_y_max
                % 计算该像素与圆心的距离
                distance = sqrt((x_center - i)^2 + (y_center - j)^2);
                
                % 如果该像素位于圆形区域内，则累加它的强度
                if distance <= r1
                    r1_intensity = r1_intensity + matrix(i, j, k);
                    r1_num_pixels = r1_num_pixels + 1;
                     mask_circle(i, j) = 1;  
                end
            end
        end

        % 保存当前圆形区域的总强度
        result(k) = r1_intensity - BG * r1_num_pixels;
        mask_ring_matrix(:, :, k) = mask_ring;  
        mask_circle_matrix(:, :, k) = mask_circle;  
    end
end





    
%     function result = sum_weighted_circular_region(matrix, x0, y0, r)
%     % matrix: 输入的矩阵
%     % x0, y0: 浮点数坐标序列
%     % r: 半径（正的有理数）
% 
%     % 输入参数检查
%     assert(r > 0, 'r 必须是正的有理数');
%     assert(length(x0) == length(y0), 'x0 和 y0 必须长度相同');
% 
%     % 初始化结果数组
%     num_points = length(x0);
%     result = zeros(1, num_points);
% 
%     % 对每对 (x0[i], y0[i]) 进行处理
%     for k = 1:num_points
%         x_center = x0(k);
%         y_center = y0(k);
%         frame = k;
% 
%         % 确定区域的边界
%         x_min = max(floor(x_center - r), 1);
%         x_max = min(ceil(x_center + r), size(matrix, 1));
%         y_min = max(floor(y_center - r), 1);
%         y_max = min(ceil(y_center + r), size(matrix, 2));
% 
%         % 初始化单个圆形区域的加权和和总权重
%         weighted_sum = 0;
%         total_weight = 0;
% 
%         % 遍历选定区域内的每个元素，计算权重并加权求和
%         for i = x_min:x_max
%             for j = y_min:y_max
%                 % 计算每个矩阵元素的边界
%                 x1 = i - 0.5;
%                 x2 = i + 0.5;
%                 y1 = j - 0.5;
%                 y2 = j + 0.5;
% 
%                 % 计算与圆的交集面积
%                 area = pixel_circle_intersection(x_center, y_center, r, x1, x2, y1, y2);
% 
%                 % 累加加权和
%                 weighted_sum = weighted_sum + matrix(i, j, k) * area;
%                 total_weight = total_weight + area;
%             end
%         end
% 
%         % 计算加权和并存储在结果数组中
%         if total_weight > 0
%             % attention
%             %result(k) = weighted_sum / total_weight;
%             result(k) = weighted_sum;
%         else
%             result(k) = 0;
%         end
%     end
% end
% 
% function area = pixel_circle_intersection(xc, yc, r, x1, x2, y1, y2)
%     % 计算矩形与圆的交集面积
%     % xc, yc: 圆心坐标
%     % r: 圆的半径
%     % x1, x2, y1, y2: 矩形的边界
% 
%     % 初始化面积
%     area = 0;
% 
%     % 检查每个小矩形的顶点是否在圆内
%     points = [x1, y1; x2, y1; x2, y2; x1, y2];
%     in_circle = sum((points(:,1) - xc).^2 + (points(:,2) - yc).^2 <= r^2);
% 
%     if in_circle == 4
%         % 如果所有顶点都在圆内，则整个矩形都在圆内
%         area = (x2 - x1) * (y2 - y1);
%     elseif in_circle > 0
%         % 如果部分顶点在圆内，则进一步计算精确的交集面积
%         % 这里可以使用更精确的积分方法或近似方法
%         % 为了简单起见，我们使用简单的估计方法
%         area = approximate_intersection_area(xc, yc, r, x1, x2, y1, y2);
%     end
% end
% 
% function area = approximate_intersection_area(xc, yc, r, x1, x2, y1, y2)
%     % 近似计算矩形与圆的交集面积
%     % 这里可以使用蒙特卡洛方法、数值积分或其他方法
%     % 为了简单起见，使用简单的估计方法
%     % 这里我们假设交集面积为矩形面积的比例
%     % 根据在圆内的顶点数目进行估计
%     area = 0;
% 
%     % 估计方法：如果部分顶点在圆内，则近似计算交集面积
%     % 这里我们使用一个简单的比例方法
%     % 创建较小的网格进行积分
%     sub_divisions = 10; % 细分数
%     dx = (x2 - x1) / sub_divisions;
%     dy = (y2 - y1) / sub_divisions;
% 
%     for xi = x1:dx:x2-dx
%         for yi = y1:dy:y2-dy
%             mid_x = xi + dx / 2;
%             mid_y = yi + dy / 2;
%             if (mid_x - xc)^2 + (mid_y - yc)^2 <= r^2
%                 area = area + dx * dy;
%             end
%         end
%     end
% end

end



