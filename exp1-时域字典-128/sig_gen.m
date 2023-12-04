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
kp=1;%导频-时延
lp=1;%导频-多普勒
delay_taps=[0];
doppler_taps=[0];
n_path=length(delay_taps);
chan_coef=[10+10*j 10+10j 10+10j];
sym_p=1;
%% 生成训练信道
left_index =[0:1:39];
right_index=[88:1:127];
separation =[1:1:40];%
num_train=16800;
chan=zeros(num_train,4);%delay1;doppler1;delay2;doppler2;ch1;ch2
chan2=rand(num_train,2)*10+5+1j*rand(num_train,2)*10+5j;

nn=1600;
lch=repmat(left_index, 1, 40);
lch2=repelem(left_index,40);
rch=repmat(right_index, 1, 40);
rch2=repelem(right_index,40);
lchs=repmat(left_index, 1, 40)+repelem(separation, 40);
rchs=repmat(right_index, 1, 40)-repelem(separation, 40);

%%%%%%时延同
chan(1:nn,1)=lch2;
chan(1:nn,2)=lch;
chan(1:nn,3)=lch2;
chan(1:nn,4)=lchs;

chan(nn+1:nn*2,1)=lch2;
chan(nn+1:nn*2,2)=rch;
chan(nn+1:nn*2,3)=lch2;
chan(nn+1:nn*2,4)=rchs;

chan(nn*2+1:nn*3,1)=rch2;
chan(nn*2+1:nn*3,2)=lch;
chan(nn*2+1:nn*3,3)=rch2;
chan(nn*2+1:nn*3,4)=lchs;

chan(nn*3+1:nn*4,1)=rch2;
chan(nn*3+1:nn*4,2)=rch;
chan(nn*3+1:nn*4,3)=rch2;
chan(nn*3+1:nn*4,4)=rchs;
%%%%%%多普勒同
chan(nn*4+1:nn*5,2)=lch2;
chan(nn*4+1:nn*5,1)=lch;
chan(nn*4+1:nn*5,4)=lch2;
chan(nn*4+1:nn*5,3)=lchs;

chan(nn*5+1:nn*6,2)=lch2;
chan(nn*5+1:nn*6,1)=rch;
chan(nn*5+1:nn*6,4)=lch2;
chan(nn*5+1:nn*6,3)=rchs;

chan(nn*6+1:nn*7,2)=rch2;
chan(nn*6+1:nn*7,1)=lch;
chan(nn*6+1:nn*7,4)=rch2;
chan(nn*6+1:nn*7,3)=lchs;

chan(nn*7+1:nn*8,2)=rch2;
chan(nn*7+1:nn*8,1)=rch;
chan(nn*7+1:nn*8,4)=rch2;
chan(nn*7+1:nn*8,3)=rchs;
%%%%%时延多普勒均不同
dleft_index=downsample(left_index,9);
dright_index=downsample(right_index,9);

dlch=repmat(dleft_index, 1, 5);
dlch=repmat(dlch,1,40);

dlch2=repelem(dleft_index,5);
dlch2=repmat(dlch2,1,40);

drch=repmat(dright_index, 1, 5);
drch=repmat(drch,1,40);

drch2=repelem(dright_index,5);
drch2=repmat(drch2,1,40);

dlchs=@(dlch) dlch+repelem(separation, 25);
drchs=@(drch) drch-repelem(separation,25);


chan(nn*8+1:nn*8+1e3,1)=dlch;
chan(nn*8+1:nn*8+1e3,2)=dlch2;
chan(nn*8+1:nn*8+1e3,3)=dlchs(dlch);
chan(nn*8+1:nn*8+1e3,4)=dlchs(dlch2);

chan(nn*8+1e3+1:nn*8+1e3*2,1)=dlch;
chan(nn*8+1e3+1:nn*8+1e3*2,2)=drch2;
chan(nn*8+1e3+1:nn*8+1e3*2,3)=dlchs(dlch);
chan(nn*8+1e3+1:nn*8+1e3*2,4)=drchs(drch2);

chan(nn*8+1e3*2+1:nn*8+1e3*3,1)=drch;
chan(nn*8+1e3*2+1:nn*8+1e3*3,2)=drch2;
chan(nn*8+1e3*2+1:nn*8+1e3*3,3)=drchs(drch);
chan(nn*8+1e3*2+1:nn*8+1e3*3,4)=drchs(drch2);

chan(nn*8+1e3*3+1:nn*8+1e3*4,1)=drch;
chan(nn*8+1e3*3+1:nn*8+1e3*4,2)=dlch2;
chan(nn*8+1e3*3+1:nn*8+1e3*4,3)=drchs(drch);
chan(nn*8+1e3*3+1:nn*8+1e3*4,4)=dlchs(dlch2);
%% 信号模型
u=@(t) 1*exp(1j*pi*B*t).*sin(pi*B*t)./(pi*B*t);
% pq= @(t,tau,v,q) u(t-tau-q'*T).*exp(1j*2*pi*v*q'*T);
% p= @(t,tau,v) sum(pq(t,tau,v,(0:N-1)),1);
sig=@(k,l,t) sum(u(t-k/B/128*M-(0:N-1)'*T).*exp(1j*2*pi*l/Tw/128*N*(0:N-1)'*T),1);%输入index,里面已经除128了
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

sig_train=zeros(num_train,N*M);
for i=1:num_train
    tmp=sig(chan(i,1),chan(i,2))*chan2(i,1);
    Y1=downsample(tmp,round(1/B/dt));
    % r1=downsample(tmp,round(1/B/dt));
    % r_mat = reshape(r1(1:N*M),M,N); 
    % Y1 = fft(r_mat)/sqrt(M); % Wigner transform
    tmp=sig(chan(i,3),chan(i,4))*chan2(i,2);
    Y2=downsample(tmp,round(1/B/dt));
    % r1=downsample(tmp,round(1/B/dt));
    % r_mat = reshape(r1(1:N*M),M,N); 
    % Y2 = fft(r_mat)/sqrt(M); % Wigner transform
    sig_train(i,:)=reshape((Y1+Y2),1,[]);%按列读取，先列后行
end
random_index=randperm(num_train);
chan=chan(random_index,:);
chan2=chan2(random_index,:);
sig_train=sig_train(random_index,:);
save('sig_gen_t.mat','sig_train','chan','chan2');


