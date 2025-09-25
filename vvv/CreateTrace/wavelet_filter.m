function [filtered, stdWavelet1] = wavelet_filter(image)

% 此函数主要用于提高单分子定位显微镜的图像质量，
% 通过小波分析方法减少噪声并强化图像中的重要特征，
% 有助于更准确的单分子检测。
% 标准差stdWavelet1可作为判定背景噪声水平的重要参数，
% 进一步用于设定适当的检测阈值，从而改善检测的灵敏度和精度。
%详见文献：
%Izeddin et al. "Wavelet analysis for single molecule localization
%microscopy", Opt. Express 20, 2081-2095 (2012) 
%
%Input:
%   image - 二维数组，表示单个图像的像素值
%
%Output:
%   filtered - 经过小波滤波处理的图像
%   stdWavelet1 - 第一小波映射的标准差，可以解释为背景噪声的度量，
%                   并用于计算检测阈值



bordObj = 4; % 设置图像边缘的像素值为0，避免边缘效应，copy from matlabtrack

% 初始化小波系数，三次样条函数不完全小波变换
H0 = 3/8;
H1 = 1/4;
H2 = 1/16;

% 滤波矩阵，
% filter_1 基于系数H0、H1、H2计算的第1个滤波器
% filter_2 基于同样的系数但加入了0间隔的第2个滤波器
filter_1 = [H2,H1,H0, H1, H2]' * [H2,H1,H0, H1, H2];
filter_2 = [H2,0,H1, 0, H0,0,H1, 0, H2]' * [H2,0,H1,0,H0,0 H1,0, H2];

if ~isa(image,'double') && ~isa(image,'single')
    image = single(image);
end

% 使用filter_1对原始图像进行卷积得到Coef1
Coef1 = conv2(image,filter_1,'same');   

% 再用filter_2对Coef1进行卷积得到Coef2
Coef2 = conv2(Coef1,filter_2,'same');   % the second coefficient map

%从原始图像中减去Coef1得到第一小波映射Wavelet1，这表示背景噪声
Wavelet1 = image - Coef1;

%计算Wavelet1的标准差stdWavelet1，用于估计背景噪声
stdWavelet1 = std(Wavelet1(:));

% Wavelet2为Coef1与Coef2的差值，代表最终的滤波图像
Wavelet2 = Coef1 - Coef2;


% 对Wavelet2的四周边界像素设置为0，避免边界附近的奇异效果
Wavelet2(1:bordObj,:) = 0;
Wavelet2((end - bordObj + 1):end,:) = 0;
Wavelet2(:,1:bordObj) = 0;
Wavelet2(:,(end - bordObj + 1):end) = 0;
Wavelet2(Wavelet2 < 0) = 0;
filtered = Wavelet2;

end