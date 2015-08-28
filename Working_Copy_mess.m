clear;
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
    RAW = RAW(2,:);
    BPM0 = BPM(1,index).BPM0;
    display(numel(BPM0));
    display(index);
    modPPG = RAW;
    %clearvars sig %%keep workspace clean
    %kill all negative values
    %subplot(4,1,1);
    %plot(RAW);
    for point = 1:numel(RAW)
        if RAW(point) <= 0
            modPPG(point) = 0;
        end
    end
    clearvars point %%keep workspace clean
    %Get the local peaks and map their locations
    %peaks returns the value of the peaks, positive and negative
    %location returns where the peak occurred
    [peaks,locations] = findpeaks(RAW);
    [peaksFilt,locationsFilt] = findpeaks(RAW,'MinPeakDistance',prominence);
    %subplot(4,1,2);
    %plot(locations,peaks);
    %xlabel('samples');
    %ylabel('sample power');
    %subplot(4,1,3);
    %plot(locationsFilt,peaksFilt);
    %map the number of locations(number of peaks) with numel
    locationCount = numel(locations);
    %%%% Create a vector of the difference between peaks in seconds %%%%%%%%%%%
    for count = 1:locationCount
        element = locations(count);
        if count < locationCount
            nextElement = locations(count+1);
        else
            nextElement = element;
        end
        nextDifferenceSeconds(count) = (nextElement - element)/125; %div 125 converts to seconds
    end
    
    count = 1;
    n = 1;
    maximum1 = numel(RAW);
    maximum2 = numel(locationsFilt);
    while count < maximum1
        if n < maximum2
            if count == locationsFilt(n)
                modPPG(count) = peaksFilt(n);
                n = n + 1;
            end
        end
        count = count + 1;
    end
    start = numel(modPPG);
    maximum = numel(RAW);
    %fix vectors to proper length
    if start < maximum
        for count = start:maximum
            modPPG(count) = RAW(count);
        end
    end
    resultSawtelle = modPPG;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
%%% SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE SAWTELLE   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
% display('COMPLETED SAWTELLE CODE');
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
% % COBO START COBO START COBO START COBO START COBO START COBO START COBO %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% display('STARTED COBO CODE');
% 
% 
% % Where you put in the current PPG.
% % Now only calls workDraftGrouping
% 
% M=1000;
% L=floor(M/2);
% K=M-L+1;
% N=4096;
% 
% fs = 125;
% windowSize=1000;
% stepSize=500;
% originalLength = length(modPPG);
% winNum = floor(originalLength/stepSize);
% 
% % Loading the bandpass filter. 
% bp = load('bpFilt.mat');
% bp = bp.bp;
% 
% % Windowing the signal, to then decompose each window.
% tempPpg = zeros(1,(winNum)*stepSize);
% tempPpg(1:originalLength) = modPPG(1:originalLength);
% windowedSignal = zeros(winNum, windowSize);
% 
% 
% % Setting up and populating the lagged Matrix. 
% % Note, all time series held in row-vectors.
% for i = 0:winNum-1 % 0-index for first row with no delay (aka window starting point)
%     for j = 1:windowSize
%         % first if catches when window went past last point.
%         if (j+(stepSize*i) > originalLength) 
%             windowedSignal(i+1, j) = 0;
%         else
%             windowedSignal(i+1,j) = tempPpg(1, j + (stepSize*i));
%         end
%     end
% end
% 
% % Will hold the different bandpassed ppgSamples.
% samples = zeros(winNum,M);
% for i = 1:winNum
%     currentWindow = filtfilt(bp, windowedSignal(i,:));
%     currentWindow = currentWindow - mean(currentWindow);
%     currentWindow = currentWindow/max(currentWindow);
%     samples(i,:) = currentWindow;
% end
% 
% % SSA DECOMPOSITION
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Grouping and restoring the oscillatory principal components to obtain
% % a restored time series periodic with the heartbeat.
% restoredWindows = zeros(winNum,windowSize);
% 
% for i = 1:winNum
%     currentWindow = workDraftGrouping(samples(i,:));
%     currentWindow = sgolayfilt(currentWindow, 0 , 19);
%     currentWindow = currentWindow-mean(currentWindow);
%     currentWindow = currentWindow/max(currentWindow);
%     restoredWindows(i,:) = currentWindow;
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% % % Restoring the PPG
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Will hold restored signal.
% resultCobo = zeros(1,winNum*stepSize);
% 
% % Chaining windows back to back.
% for i = 0:winNum-1
%     currentWindow = restoredWindows(i+1, :);
%     resultCobo( 1, 1 + i*stepSize : M + i*stepSize ) = currentWindow;
% end
% 
% if length(resultCobo) > start
%     resultCobo = resultCobo(1:start);
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
% %%%%%% COBO END COBO END COBO END COBO END COBO END COBO END COBO %%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% display('COMPLETED COBO CODE');

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
%x=resultCobo;
x=modPPG;
peaks =zeros;
yg=zeros;
peaks_ac =zeros;
collp = zeros;
error_peaks = zeros;
peaks1 = zeros; 
track =zeros; 
error_peaks1 = zeros;
max_aa = zeros; 

%filter: 
OR=2; %Order
fc1= 1.4 %1st Frequency cutoff
fc2 =2.3 %2nd Freq. cutoff
[b,a] = butter(OR, [fc1 fc2]/(fs/2), 'bandpass');
x1 = filter(b,a,x);

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
%ac=x2;
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
win = 0.1;
if (n>4)
    if (max_aa(end) < win & max_aa(end-1) < win)
        delta=delta+6;
    end
    if (max_aa(end) < win & max_aa(end-1) < win & max_aa(end-2) < win)
        delta=delta+8;
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

%%%%%%%%%%%%%%%%%%%%%discriminate against outliers%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  if (length(peaks1)>4)
%  peaks1(end)=(peaks1(end)*.25*(peaks1(end-1)+peaks1(end-2)+peaks1(end-3))/3);
%  end

peaksize=length(peaks1);
% window = 10;
% if (peaksize > window)
% suspects=peaks1(end-window:end-1);
% length(suspects);
% suspects=track((end-length(suspects)+1):end).*suspects(1:end);
% nz=find(suspects);
% suspects=suspects(nz);
% suspects1=sum(suspects)/length(suspects);
% 
% if (peaks1(end) > suspects1+0.3*(suspects1) | peaks1(end) < suspects1-0.3*(suspects1)) %???
%     peaks1(end) = suspects1;
%      peaks = peaks1(end-peaksize+1:end);
%     % peaks = vertcat(peaks,60*f1(peak_a));
%      k=linspace(1.0,peaksize,peaksize)
%      p= polyfit(k,peaks,2);
%      yguess=polyval(p,k);
%      BPM0(n,1);
%      %yg=[yg,yguess(end)];
%      peaks1(end)=yguess(end)
%     %If peaks have been used ignore them. 
%      track(n)=0;
%     
% end
% end %end for window size
% collect peak values for poly fit
% peaksize=length(peaks); 
% if (peaks(1)==0)
%     peaks = 60*f(peak_a);
%     %peaksize=peaksize+1;
% end
% if (peaksize(1) == 3)
%     peaks = peaks(2:end);
%     peaks = vertcat(peaks,60*f(peak_a));
% else
% peaks=vertcat(peaks,60*f(peak_a));
% end 
% %k=linspace(1.0,peaksize,peaksize+1)
% if (peaksize(1) <3)
%     k=1:peaksize(1)+1;
% end
% %polynomial fitting:
% peaks;
% size(k);
% k;
% size(peaks');
% BPM0(n,1);
% p= polyfit(k,peaks',1);
% yguess=polyval(p,k);
% BPM0(n,1);
% yg=[yg,yguess(end)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %Check for flag condition
flag20= abs(peaks1(end)-actual);
if (flag20 > 20)
   datasetfailed = 1;
end

%smooth 


peaksize=length(peaks1);
%Collecting and estimating error: 
error_acp=(abs(60*f1(peak_a)-BPM0(n,1))/BPM0(n,1))*100;
error_ac=[error_ac error_acp];
peaks_ac= [peaks_ac abs(60*f1(peak_a))];
error_peaks =[error_peaks ((peaks(end)-BPM0(n,1))/BPM0(n,1))*100];
error_peaks1 =[error_peaks1 abs(((peaks1(end)-BPM0(n,1))/BPM0(n,1))*100)];
hold off;pause(.01);
end %while L > 0
%peaks1=smooth(peaks1,0.05, 'rloess');
avg_peaks1_error = (sum(error_peaks1)/length(error_peaks1))
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
if datasetfailed == 1
    %Error(index,2) = -1;
end
pause(1);
end
Error(14,2) = mean(Error(1:13,2));
toc