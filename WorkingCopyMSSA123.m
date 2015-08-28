    % clear;
tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
%%% SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Access BPM(1,1).BPM0
%Access PPG(1,1).sig
Error = zeros(14,2);
count = 1;
Error(14,1) = 0;
for count = 1:14
   Error(count,1) = count;
end

index = 1;
clearvars count;

for index = 1:13
    display('Starting');
    display(index);
    current = pwd;
    path1 = '/PPG/PPG';
    path2 = '/BPM/Trace';
    PPGpath = strcat(current,path1,num2str(index));
    BPMpath = strcat(current,path2,num2str(index));
    PPG(index) = load(PPGpath);
    BPM(index) = load(BPMpath);
    display(PPGpath);
    display(BPMpath);
    prominence = 85;
    sample = 1;
    RAW = PPG(1,index).sig;
    % MSSA Using both ppg channels.
    RAW1 = RAW(2,:);
    RAW2 = RAW(3,:);
    BPM0 = BPM(1,index).BPM0;
    display(numel(BPM0));
    display(index);
    modPPG1 = RAW1;
    modPPG2 = RAW2;
    %clearvars sig %%keep workspace clean
    %kill all negative values
    %subplot(4,1,1);
    %plot(RAW);
    for point = 1:numel(RAW1)
        if RAW1(point) <= 0
            modPPG1(point) = 0;
        end
    end
    for point = 1:numel(RAW2)
        if RAW2(point) <= 0
            modPPG2(point) = 0;
        end
    end
    clearvars point %%keep workspace clean
    %Get the local peaks and map their locations
    %peaks returns the value of the peaks, positive and negative
    %location returns where the peak occurred

    [peaksFilt1,locationsFilt1] = findpeaks(RAW1,'MinPeakDistance',prominence);
    [peaksFilt2,locationsFilt2] = findpeaks(RAW2,'MinPeakDistance',prominence);
    
    %subplot(4,1,2);
    %plot(locations,peaks);
    %xlabel('samples');
    %ylabel('sample power');
    %subplot(4,1,3);
    %plot(locationsFilt,peaksFilt);
    %map the number of locations(number of peaks) with numel
    locationCount1 = numel(locations1);
    locationCount2 = numel(locations2);
    %%%% Create a vector of the difference between peaks in seconds %%%%%%%%%%%
    for count = 1:locationCount1
        element = locations1(count);
        if count < locationCount1
            nextElement = locations1(count+1);
        else
            nextElement = element;
        end
        nextDifferenceSeconds1(count) = (nextElement - element)/125; %div 125 converts to seconds
    end
    
    for count = 1:locationCount2
        element = locations2(count);
        if count < locationCount2
            nextElement = locations2(count+1);
        else
            nextElement = element;
        end
        nextDifferenceSeconds2(count) = (nextElement - element)/125; %div 125 converts to seconds
    end
    
    % Applying Chris S filt to 1st ppg signal.
    count1 = 1;
    n1 = 1;
    maximum1 = numel(RAW1);
    maximum2 = numel(locationsFilt1);
    while count1 < maximum1
        if n1 < maximum2
            if count1 == locationsFilt1(n1)
                modPPG1(count1) = peaksFilt1(n1);
                n1 = n1 + 1;
            end
        end
        count1 = count1 + 1;
    end
    start1 = numel(modPPG1);
    maximumA = numel(RAW1);
    %fix vectors to proper length
    if start1 < maximumA
        for count1 = start1:maximumA
            modPPG1(count1) = RAW1(count1);
        end
    end
    resultSawtelle1 = modPPG1;
    
    % Applying Chris filt to 2nd ppg signal.
    count2 = 1;
    n2 = 1;
    maximum1 = numel(RAW2);
    maximum2 = numel(locationsFilt2);
    while count2 < maximum1
        if n2 < maximum2
            if count2 == locationsFilt2(n2)
                modPPG2(count2) = peaksFilt2(n2);
                n2 = n2 + 1;
            end
        end
        count2 = count2 + 1;
    end
    start2 = numel(modPPG2);
    maximumB = numel(RAW2);
    %fix vectors to proper length
    if start2 < maximumB
        for count2 = start2:maximumB
            modPPG2(count2) = RAW2(count2);
        end
    end
    resultSawtelle2 = modPPG2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
%%% SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
display('COMPLETED SAWTELLE CODE');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
% COBO START COBO START COBO START COBO START COBO START COBO START COBO %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('STARTED COBO CODE');

windowSize=1000;
stepSize=500; % robut for 250-625, but gets noticeably slower with smaller
              % step size.
L=floor(windowSize/2);
K=windowSize-L+1;
N=4096;
fs = 125;


% Setting original length and window number of each signal.
originalLength1 = length(modPPG1);
originalLength2 = length(modPPG2);
% Should be the same, take min to handle length issue.
winNum1 = floor(originalLength1/stepSize);
winNum2 = floor(originalLength2/stepSize);

winNum = min(winNum1,winNum2);

% Windowing the signal to decompose each window.
% Will not be needed when input is single window.
tempPpg1 = zeros(1,(winNum)*stepSize);
tempPpg1(1:originalLength1) = modPPG1(1:originalLength1);
tempPpg2 = zeros(1,(winNum)*stepSize);
tempPpg2(1:originalLength2) = modPPG2(1:originalLength2);

windowedSignal1 = zeros(winNum, windowSize);
windowedSignal2 = zeros(winNum, windowSize);

% Setting up and populating the lagged Matrix. 
% Note, all time series held in row-vectors.
for i = 0:winNum-1 % 0-index for first row with no delay (aka window starting point)
    for j = 1:windowSize
        if (j+(stepSize*i) > originalLength1) 
            windowedSignal1(i+1, j) = 0;
            windowedSignal2(i+1, j) = 0;
        else
            windowedSignal1(i+1,j) = tempPpg1(1, j + stepSize*i);
            windowedSignal2(i+1,j) = tempPpg2(1, j + stepSize*i);
        end
    end
end

% Will hold the different ppg windows for each.
samples1 = zeros(winNum,windowSize);
samples2 = zeros(winNum,windowSize);
for i = 1:winNum
 
    currentWindow1 = sgolayfilt(windowedSignal1(i,:), 0 , 19);
    currentWindow1 = currentWindow1 - mean(currentWindow1);
    currentWindow1 = currentWindow1/max(currentWindow1);
    samples1(i,:) = currentWindow1;
    
    currentWindow2 = sgolayfilt(windowedSignal2(i,:), 0 , 19);
    currentWindow2 = currentWindow2 - mean(currentWindow2);
    currentWindow2 = currentWindow2/max(currentWindow2);
    samples2(i,:) = currentWindow2;
end

% VMSSA DECOMPOSITION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Grouping and restoring the oscillatory principal components to obtain
% a restored time series periodic with the heartbeat.
restoredWindows = zeros(winNum,windowSize);

for i = 1:winNum
    currentWindow = groupingMSSA(samples1(i,:), samples2(i,:));
    currentWindow = sgolayfilt(currentWindow, 0 , 19);
    currentWindow = currentWindow-mean(currentWindow);
    currentWindow = currentWindow/max(currentWindow);
    restoredWindows(i,:) = currentWindow;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Restoring the PPG
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

resultCobo = zeros(1,winNum*stepSize);

% Chaining windows back to back.
for i = 0:winNum-1
    currentWindow = restoredWindows(i+1, :);
    resultCobo(1 + i*stepSize : windowSize + i*stepSize ) = currentWindow;
end
if length(resultCobo) > start1
    resultCobo = resultCobo(1:start1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
%%%%%% COBO END COBO END COBO END COBO END COBO END COBO END COBO %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('COMPLETED COBO CODE');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% WELCH START WELCH START WELCH START WELCH START WELCH START WELCH %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('STARTED WELCH CODE');


%The purpose of this script to to study the ability of 
%straight data FFT, and autocorrelation's to match that of the
% average heartbeat recorded via BPMTrace. 
error_og = zeros;
error_ac = zeros;
%data: 

%parameters
fs=125;
fmin=1;
fmax=3.3;

%signal
x=resultCobo;
peaks =zeros;
yg=zeros;
peaks_ac =zeros;
collp = zeros;
error_peaks = zeros;
peaks1 = zeros; 
track =zeros; 
error_peaks1 = zeros;
max_aa = zeros; 

%Loop counter variables. (SECONDS NOT DATA POINTS
L=length(x1(1,:))-(8*fs) %Because first average point is after 8 seconds.
N=(L/(2*fs)) %number of 2 second periods left. 
firstrun=1;
datasetfailed = 0;

while N > 1 %Need it to be 1 to go to end without error?
if firstrun ==1; 
    x2=x1(1,1:(8*fs));
    last=8*fs;
    firstrun = 0;
    n=1;
else 
    last=last+2*fs;
    x2=x1(1,(last-8*fs):last);
    
    %N = N-2; wtf??? This was a big mistake.... ( .__.)
    N=N-1;
    n=n+1;
end

    n;
%analysis
ac=xcorr(x2);
t1=0:(1/125):(length(ac)-1)/125;
t=0:(1/125):(length(x2)-1)/125;

%plot auto correlation
subplot(1,2,1);
plot(t1,ac);
%axis([8 16 -2e5 4e5]);
title('Autocorrelation fucntion of the PPG');
xlabel('lag in seconds');
ylabel('Autocorrelation');

%Take FFT of auto correlation
%NFFT=2*(2^nextpow2(length(ac))); %Next power of 2
NFFT=4096;
f=fs/2*linspace(0,1,NFFT/2+1);
%f=pi*(f/max(f));
%FFT and Normalization over an interval: 
Y=fft(ac, NFFT)/(length(x2)); % Take fft take
Y=2*abs(Y(1:NFFT/2+1)); %Take care of 2 factor, and absolute
roii=fmax>f & f>fmin; %Setup frequency range that is realistic as well. 
Y=Y(roii)/max(Y(roii)); %Set specific region up, and normalize fft. 
f1=f(roii);
%
%Plot autocorrelation FFT
subplot(1,2,2);
roii=fmax>f & f>fmin;
plot(60*f1, Y);
hold on;
grid on
title('FFT of autocorrelation function');
xlabel('BPM');
ylabel('Amp.');

%Below is the variable window size limits: 
delta=6;
pdm = 1; %default: peak loctioan minus delta equals zero. 
pdp = length(f1); %default: peak loctioan plus delta equals end of spectrum

%Variable window size:
% if peak was below 10% maximum, increase window
% if below 10% again, double window. 
win = 0.05;
if (n>4)
    if (max_aa(end) < win & max_aa(end-1) < win)
        delta=delta+2;
    end
    if (max_aa(end) < win & max_aa(end-1) < win & max_aa(end-2) < win)
        delta=delta+4;
    end
end


% add peak fft value
 n;
 if (n<2)
roi=1.98>f1 & f1>1.0;
 else
     pdp=peak_a+delta;
     pdm=peak_a-delta;
 end
 
 if(pdp > length(f1))
     pdm=pdm-(abs(pdp)-length(f1));
     pdp = length(f1);
 end
 if (pdm <1)
     pdp=pdp+abs(pdm);
     if (pdp >length(f1))
         pdp = length(f1);
     end
     pdm =1;
 end
 if (n>2)
     pdp;
     pdm;
     roi = f1(pdp)>f1 & f1>f1(pdm);
 end
%Find index of highest peak within search window
peak_a=find(Y(:) == max(Y(roi)));
%max value (amplitude)
max_a=Y(peak_a);
max_aa=[max_aa max_a];
%Plot guess: 
stem(60*f1(peak_a),max_a);

% % %!! Testing acc noise removal.
% % % Grabbing previous estimate to pass into grouping.
% % % prevEst = peak_a;

% beast=60*f1(peak_a)

%find current actual value
actual=BPM0(n,1);

%Plot search window: 
hold on;
plot([60*f1(pdm) 60*f1(pdm)], get(gca,'ylim'));
hold on;
plot([60*f1(pdp) 60*f1(pdp)], get(gca,'ylim'));
hold on; 
%Plot actual
stem(BPM0(n,1),max_a, 'r');
hold on;
track(n)=1;

%Collect all peaks locations
peaks1 = [peaks1, 60*f1(peak_a)];
flag20= abs(peaks1(end)-actual);
% if (flag20 > 20)
%    datasetfailed = 1;
% 
% end

%smooth 

peaksize=length(peaks1);
%Collecting and estimating error: 
error_acp=(abs(60*f1(peak_a)-BPM0(n,1))/BPM0(n,1))*100;
error_ac=[error_ac error_acp];
peaks_ac= [peaks_ac abs(60*f1(peak_a))];
error_peaks =[error_peaks ((peaks(end)-BPM0(n,1))/BPM0(n,1))*100];
error_peaks1 =[error_peaks1 abs(((peaks1(end)-BPM0(n,1))/BPM0(n,1))*100)];
hold off;pause(.0001);
end %while L > 0

peaks1=smooth(peaks1,0.05, 'rloess');
avg_peaks1_error = (sum(error_peaks1)/length(error_peaks1));
avg_error_ac=(sum(error_ac)/length(error_ac))
%figure
plot(BPM0(:,1));
hold on; 
plot(peaks_ac, 'r');
hold on
plot(peaks1, 'g');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% WELCH END WELCH END WELCH END WELCH END WELCH END WELCH %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('COMPLETED WELCH CODE');
display(datasetfailed);
Error(index,2) = avg_error_ac;
% if datasetfailed == 1
%     Error(index,2) = worstError;
% end
end
Error(14,2) = mean(Error(1:13,2));
toc