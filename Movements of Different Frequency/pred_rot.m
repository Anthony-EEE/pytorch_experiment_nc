% Copyright Tianchen Shen, Irene Di Giulio, Matthew Howard. 
% This code is provided confidentially for purposes of peer review only. 
% All rights reserved.


clc
clear
% clc

addpath 'C:/Users/11424/Documents/phd/Movements of Different Frequency'
folder1 = 'C:/Users/11424/Documents/phd/Movements of Different Frequency/pwm_37';
folder2 = 'C:/Users/11424/Documents/phd/Movements of Different Frequency/pwm_40';
repeats =50;
rep_time=100;
max_iter=100;
thresh=1e-5;
t = 100;
%% 
class1 = importdata([folder1, '/1_000.csv']);
class2 = importdata([folder2, '/1_000.csv']);
s1(:,1) = [9;21;33;4];  %x -axis
s1(:,2) = [10;22;34;5]; %y -axis
s1(:,3) = [11;23;35;6]; %z -axis
s2(:,1) = [6;18;30;1];  %x -axis
s2(:,2) = [7;19;31;2]; %y -axis
s2(:,3) = [8;20;32;3]; %z -axis

%%
for i = 1 : 3
tclass1_x(:,i)= str2double(class1.textdata(2:end,s1(i,1)));
tclass1_y(:,i)= str2double(class1.textdata(2:end,s1(i,2)));
tclass1_z(:,i)= str2double(class1.textdata(2:end,s1(i,3)));

rclass1_x(:,i)= str2double(class1.textdata(2:end,s2(i,1)));
rclass1_y(:,i)= str2double(class1.textdata(2:end,s2(i,2)));
rclass1_z(:,i)= str2double(class1.textdata(2:end,s2(i,3)));

tclass2_x(:,i)= str2double(class2.textdata(2:end,s1(i,1)));
tclass2_y(:,i)= str2double(class2.textdata(2:end,s1(i,2)));
tclass2_z(:,i)= str2double(class2.textdata(2:end,s1(i,3)));

rclass2_x(:,i)= str2double(class2.textdata(2:end,s2(i,1)));
rclass2_y(:,i)= str2double(class2.textdata(2:end,s2(i,2)));
rclass2_z(:,i)= str2double(class2.textdata(2:end,s2(i,3)));

end

%%
tclass1_x(:,4)= class1.data(1:end,s1(4,1));
tclass1_y(:,4)= class1.data(1:end,s1(4,2));
tclass1_z(:,4)= class1.data(1:end,s1(4,3));

rclass1_x(:,4)= class1.data(1:end,s2(4,1));
rclass1_y(:,4)= class1.data(1:end,s2(4,2));
rclass1_z(:,4)= class1.data(1:end,s2(4,3));

tclass2_x(:,4)= class2.data(1:end,s1(4,1));
tclass2_y(:,4)= class2.data(1:end,s1(4,2));
tclass2_z(:,4)= class2.data(1:end,s1(4,3));

rclass2_x(:,4)= class2.data(1:end,s2(4,1));
rclass2_y(:,4)= class2.data(1:end,s2(4,2));
rclass2_z(:,4)= class2.data(1:end,s2(4,3));

%% fill missing data
    for i = 1 : 4
index = find(tclass1_x(:,i) == -3.697314000000000e+28);
tclass1_x(index,i)= NaN;
tclass1_x(:,i) = fillmissing(tclass1_x(:,i),'spline');

index = find(tclass1_y(:,i) == -3.697314000000000e+28);
tclass1_y(index,i)= NaN;
tclass1_y(:,i) = fillmissing(tclass1_y(:,i),'spline');

index = find(tclass1_z(:,i) == -3.697314000000000e+28);
tclass1_z(index,i)= NaN;
tclass1_z(:,i) = fillmissing(tclass1_z(:,i),'spline');


index = find(rclass1_x(:,i) == -3.697314000000000e+28);
rclass1_x(index,i)= NaN;
rclass1_x(:,i) = fillmissing(rclass1_x(:,i),'spline');

index = find(rclass1_y(:,i) == -3.697314000000000e+28);
rclass1_y(index,i)= NaN;
rclass1_y(:,i) = fillmissing(rclass1_y(:,i),'spline');

index = find(rclass1_z(:,i) == -3.697314000000000e+28);
rclass1_z(index,i)= NaN;
rclass1_z(:,i) = fillmissing(rclass1_z(:,i),'spline');



index = find(tclass2_x(:,i) == -3.697314000000000e+28);
tclass2_x(index,i)= NaN;
tclass2_x(:,i) = fillmissing(tclass2_x(:,i),'spline');

index = find(tclass2_y(:,i) == -3.697314000000000e+28);
tclass2_y(index,i)= NaN;
tclass2_y(:,i) = fillmissing(tclass2_y(:,i),'spline');

index = find(tclass2_z(:,i) == -3.697314000000000e+28);
tclass2_z(index,i)= NaN;
tclass2_z(:,i) = fillmissing(tclass2_z(:,i),'spline');


index = find(rclass2_x(:,i) == -3.697314000000000e+28);
rclass2_x(index,i)= NaN;
rclass2_x(:,i) = fillmissing(rclass2_x(:,i),'spline');

index = find(rclass2_y(:,i) == -3.697314000000000e+28);
rclass2_y(index,i)= NaN;
rclass2_y(:,i) = fillmissing(rclass2_y(:,i),'spline');

index = find(rclass2_z(:,i) == -3.697314000000000e+28);
rclass2_z(index,i)= NaN;
rclass2_z(:,i) = fillmissing(rclass2_z(:,i),'spline');
end
%%
for i = 1:size(rclass1_x, 2)
    delta = 0 - rclass1_x(1, i); 
    rclass1_x(:, i) = rclass1_x(:, i) + delta; 
    rclass2_x(:, i) = rclass2_x(:, i) + delta; 

end


for i = 1:size(rclass1_y, 2)
    delta = 0 - rclass1_y(1, i); 
    rclass1_y(:, i) = rclass1_y(:, i) + delta; 
    rclass2_y(:, i) = rclass2_y(:, i) + delta; 

end


for i = 1:size(rclass1_z, 2)
    delta = 0 - rclass1_z(1, i); 
    rclass1_z(:, i) = rclass1_z(:, i) + delta; 
    rclass2_z(:, i) = rclass2_z(:, i) + delta; 
   
end
%% start time
diff_Tz = diff(tclass2_x(400:end,1));
start_indices = find(abs(diff_Tz) >1);
refined_start_indices_class2 = start_indices(1);
for i = 2:length(start_indices)
    if start_indices(i) - refined_start_indices_class2(end) >= 680
        refined_start_indices_class2 = [refined_start_indices_class2; start_indices(i)];
    end
end

diff_Tz = diff(tclass1_x(400:end,1));
start_indices = find(abs(diff_Tz) > 1);
refined_start_indices_class1 = start_indices(1);
for i = 2:length(start_indices)
    if start_indices(i) - refined_start_indices_class1(end) >= 680
        refined_start_indices_class1 = [refined_start_indices_class1; start_indices(i)];
    end
end
refined_start_indices_class1=refined_start_indices_class1+399;
refined_start_indices_class2=refined_start_indices_class2+399;

%%
for i = 1 : repeats
[max_class1(i),start_class1(i)] = max(tclass1_x(refined_start_indices_class1(i):refined_start_indices_class1(i)+200,1));
[max_class2(i),start_class2(i)] = max(tclass2_x(refined_start_indices_class2(i):refined_start_indices_class2(i)+200,1));
end


start1_class1=start_class1+refined_start_indices_class1';
start1_class2=start_class2+refined_start_indices_class2';


%% generate train_num and test_num
train=[];
test=[];
num=1;
while(num<201)
A = [1:repeats];   
A_temp = A;
% test
while length(train)<floor(repeats-1)
    temp = 1+floor(repeats.*rand(1));
    if ismember(temp,A_temp)
        train = [train, temp];
        A_temp = setdiff(A_temp,temp);
    end
end
 
% train
test= A_temp;
 
% sort()
train_num(:,num)= sort(train);
test_num(:,num) = sort(test);
num=num+1;
train=[];
test=[];
end


%%
for f = 1 : 4 %sensor
data=[];
     for i = 1 : repeats
 data(1,:,i) =tclass1_y(start1_class1(i):start1_class1(i)+t,f);
 data(2,:,i) =tclass1_z(start1_class1(i):start1_class1(i)+t,f);
 data(3,:,i) =tclass1_x(start1_class1(i):start1_class1(i)+t,f);

     
 data(4,:,i) =tclass2_y(start1_class2(i):start1_class2(i)+t,f);
 data(5,:,i) =tclass2_z(start1_class2(i):start1_class2(i)+t,f);
 data(6,:,i) =tclass2_x(start1_class2(i):start1_class2(i)+t,f);


 data(7,:,i) =rclass1_y(start1_class1(i):start1_class1(i)+t,f);
 data(8,:,i) =rclass1_z(start1_class1(i):start1_class1(i)+t,f);
 data(9,:,i) =rclass1_x(start1_class1(i):start1_class1(i)+t,f);

     
 data(10,:,i) =rclass2_y(start1_class2(i):start1_class2(i)+t,f);
 data(11,:,i) =rclass2_z(start1_class2(i):start1_class2(i)+t,f);
 data(12,:,i) =rclass2_x(start1_class2(i):start1_class2(i)+t,f);

  end

t1 = [data(7,:,:);data(8,:,:);data(9,:,:)];
t2 = [data(10,:,:);data(11,:,:);data(12,:,:)];
parfor j = 1:rep_time
        fprintf('running at: sensor %d, trial num: %d\n',f,j)
O = 3; %Number of coefficients in a vector
T = size(data,2); %Number of vectors in a sequence
nex =repeats-1; %Number of sequences
Q = T; %Number of states
M=1;
cov_type = 'full';
%%
prior0_t1 = normalise(rand(Q,1));
transmat0_t1 = mk_stochastic(rand(Q,Q));
[mu0_t1, Sigma0_t1] = mixgauss_init(Q*M, t1(:,:,[train_num(:,j+1)]), cov_type,'rnd');
mu0_t1 = reshape(mu0_t1, [O Q M]);
Sigma0_t1 = reshape(Sigma0_t1, [O O Q M]);
           mixmat0_t1 = []; % if emission probability is single Gaussian distribtuion (M=1)
%        mixmat0_t1 = mk_stochastic(rand(Q,M));

%%
prior0_t2 = normalise(rand(Q,1));
transmat0_t2 = mk_stochastic(rand(Q,Q));
[mu0_t2, Sigma0_t2] = mixgauss_init(Q*M, t2(:,:,[train_num(:,j)]), cov_type,'rnd');
mu0_t2 = reshape(mu0_t2, [O Q M]);
Sigma0_t2 = reshape(Sigma0_t2, [O O Q M]);
           mixmat0_t2 =[];% if emission probability is single Gaussian distribtuion (M=1)
%        mixmat0_t2 = mk_stochastic(rand(Q,M));

 %% t1 frequency

[LL_t1, prior1_t1, transmat1_t1, mu1_t1, Sigma1_t1, mixmat1_t1] = ...
         mhmm_em(t1(:,:,[train_num(:,j+1)]), prior0_t1, transmat0_t1, mu0_t1, Sigma0_t1, mixmat0_t1, 'max_iter', max_iter,'verbose', 0,'thresh',thresh);

%% t2 frequency
[LL_t2, prior1_t2, transmat1_t2, mu1_t2, Sigma1_t2, mixmat1_t2] = ...
         mhmm_em(t2(:,:,[train_num(:,j)]), prior0_t2, transmat0_t2, mu0_t2, Sigma0_t2, mixmat0_t2, 'max_iter', max_iter,'verbose', 0,'thresh',thresh);

%% 
for i = 1 : t
loglik_t1(i) = mhmm_logprob(t2(:,1:i,[test_num(:,j)]), prior1_t1, transmat1_t1, mu1_t1, Sigma1_t1, mixmat1_t1);
loglik_t2(i) = mhmm_logprob(t2(:,1:i,[test_num(:,j)]), prior1_t2, transmat1_t2, mu1_t2, Sigma1_t2, mixmat1_t2);
 diff_t1(f,j,i) = loglik_t2(i)-loglik_t1(i);
end
loglik_t1=[];
loglik_t2=[];

for i = 1 : t
loglik_t1(i) = mhmm_logprob(t1(:,1:i,[test_num(:,j+1)]), prior1_t1, transmat1_t1, mu1_t1, Sigma1_t1, mixmat1_t1);
loglik_t2(i) = mhmm_logprob(t1(:,1:i,[test_num(:,j+1)]), prior1_t2, transmat1_t2, mu1_t2, Sigma1_t2, mixmat1_t2);
 diff_t2(f,j,i) = loglik_t1(i)-loglik_t2(i);
end
loglik_t1=[];
loglik_t2=[];
end

end
for i = 1:t
    for j = 1 : 4
acc_low(i,j)=length(find(diff_t1(j,:,i)>0))/rep_time;
acc_high(i,j)=length(find(diff_t2(j,:,i)>0))/rep_time;
    end
end

acc=(acc_low+acc_high)/2;
distance = diff_t1+diff_t2;

  save(num2str(name))
