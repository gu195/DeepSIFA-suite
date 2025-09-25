function result = sample(filename, workdir, parameters)
%clc
%clear
%workdir = 'C:\Users\zhou-\Desktop\matlab2cpp';
%filename = 'H:\Img_Proc\motion_img\film2.tif';

parameter = struct();
parameter.nStart = parameters.frameStart;
parameter.nEnd = parameters.frameEnd - 1;


parameter.threshold = parameters.threshold;
parameter.spotTrackingRadius = parameters.spotTrackingRadius;
parameter.gaussFitWidth = parameters.gaussFitWidth;

parameter.frameLength = parameters.frameLength;
parameter.frameGap = parameters.frameGap;

parameter.trackMethod = parameters.trackMethod;%'u-track' or 'default'
parameter.utrackMotionType = parameters.utrackMotionType;
mainWithoutUI(filename, workdir, parameter);
result = 1; %parameter.trackMethod;
end




