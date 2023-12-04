clear;clc
%% 参数
M=8;
N=8;
B=2e6;%2M Hz
delta_f=B/M;
T=1/delta_f;
Ts=1/B;
Tw=N*T;
fs=1e9;
dt=1/fs;
num_all=round(Tw/dt);
t=(1:num_all)*dt+eps;%为了正确算极限
%% 生成测试信道
num_test=5000;
max_ch=4*2;
max_delay=10;
max_doppler=8;
chan=ones(num_test,max_ch).*128;%如果是128，表示没这个信道
chan2=rand(num_test,4)*10+5+1j*rand(num_test,4)*10+5j;
for i=1:num_test
    for j=1:randi(4)
        chan(i,(j-1)*2+1)=randi(max_delay);
        chan(i,(j-1)*2+2)=randi(max_doppler);
    end
end

%% 信号模型
u=@(t) 1*exp(1j*pi*B*t).*sin(pi*B*t)./(pi*B*t);
% pq= @(t,tau,v,q) u(t-tau-q'*T).*exp(1j*2*pi*v*q'*T);
% p= @(t,tau,v) sum(pq(t,tau,v,(0:N-1)),1);
sig=@(k,l) sum(u(t-k/B/128*M-(0:N-1)'*T).*exp(1j*2*pi*l/Tw/128*N*(0:N-1)'*T),1);%输入index,里面已经除128了
%% 信道/回波
% for i=1:n_path
%     taui=delay_taps(i)/B;
%     vi=doppler_taps(i)/Tw;
%     out=out+chan_coef(i)*sym_p*p(t-taui,kp/B,lp/Tw).*exp(1j*2*pi*vi*t);
%     %p(t-taui,kp/B,lp/Tw).*exp(1j*2*pi*vi*t)-p(t,kp/B-taui,lp/Tw+vi)=0
% end
% r1=downsample(out,round(1/B/dt));
% r_mat = reshape(r1(1:N*M),M,N); 
% Y = fft(r_mat)/sqrt(M); % Wigner transform

sig_test=zeros(num_test,N*M);
for i=1:num_test
    for j=1:4
        re=0;
        if(chan(i,j)==128)
            break
        else
            tmp=sig(chan(i,(j-1)*2+1),chan(i,(j-1)*2+2))*chan2(i,j);
            Y1=downsample(tmp,round(1/B/dt));
            re=re+Y1;
        end
    end
    sig_test(i,:)=reshape(re,1,[]);%按列读取，先列后行
end
random_index=randperm(num_test);
chan=chan(random_index,:);
chan2=chan2(random_index,:);
sig_test=sig_test(random_index,:);
save('sig_test_t.mat','sig_test','chan','chan2');


