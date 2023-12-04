clear;
%% 生成字典矩阵
% torch是左乘
% 离散时频域
% A=[];
% for k=0:127
%     for l=0:127
%         a1=exp(-1j*2*pi*(k)/128*(0:8-1)');
%         a2=exp(1j*2*pi*(l)/128*(0:8-1)');
%         a_est=kron(a2,a1);
%         A=[A,a_est];
%     end
% end
% A_real=real(A);
% A_imag=imag(A);
% save('A.mat',"A_imag","A_real");
%% 时域字典
M=8;
N=8;
B=2e6;%2M Hz
delta_f=B/M;
T=1/delta_f;
Ts=1/B;
Tw=N*T;
fs=1e8;
dt=1/fs;
num_all=round(Tw/dt);
t=(1:num_all)*dt+eps;%为了正确算极限
u=@(t) 1*exp(1j*pi*B*t).*sin(pi*B*t)./(pi*B*t);
% pq= @(t,tau,v,q) u(t-tau-q'*T).*exp(1j*2*pi*v*q'*T);
% p= @(t,tau,v) sum(pq(t,tau,v,(0:N-1)),1);
sig=@(k,l) sum(u(t-k/B/32*M-(0:N-1)'*T).*exp(1j*2*pi*l/Tw/32*N*(0:N-1)'*T),1);%输入index,里面已经除128了
A=[];
for k=0:31
    for l=0:31
        tmp=sig(k,l);
        a_est=downsample(tmp,round(1/B/dt));
        A=[A;a_est];
    end
end
A=conj(A);
save('A_t.mat',"A");