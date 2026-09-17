function results = tracking_experiment_qblms(mode)
% Reproduce Eq. (5), with overlapping blocks updated every sample.
% Run: qblms_tracking_experiment
% Exact numerical check: qblms_tracking_experiment('verify')
% The default uses the frozen parameters from results.json.
if nargin < 1, mode = 'experiment'; end
root = fileparts(mfilename('fullpath'));

if strcmp(mode,'verify')
    S = load(fullfile(root,'matlab_verification.mat'));
    R = size(S.xs,1); C = numel(S.qs); B = numel(S.cuts)-1;
    obtained = zeros(R,C,B,2);
    for r = 1:R
        hstates = squeeze(S.hs(r,:,:)).';
        obtained(r,:,:,:) = one_run(S.xs(r,:),S.ds(r,:),hstates,S.states,...
            S.qs,S.mus,S.leakmode,S.cuts);
    end
    maxError = max(abs(obtained(:)-S.expected(:)));
    fprintf('MATLAB vs Python: max absolute metric difference = %.12g\n',maxError);
    assert(maxError < 1e-9,'MATLAB/Python verification failed.');
    results = maxError;
    return
end

cfg = jsondecode(fileread(fullfile(root,'results.json')));
C = numel(cfg.results);
qs = zeros(1,C); etas = qs; leakmode = qs; names = strings(1,C);
for c=1:C
    qs(c) = cfg.results(c).q; etas(c) = cfg.results(c).eta;
    names(c) = string(cfg.results(c).label);
end
leakmode(end)=1;
R=200; T=12000; M=120; K=5; burn=2000; interval=1000;
h = fir1(M-1,0.4).';  % Signal Processing Toolbox
states = mod(floor((0:T-1)/interval),2);
cuts=0:100:T; perRun=zeros(R,C,2); curves=zeros(C,numel(cuts)-1,2);

for r=1:R
    rng(1910000+r,'twister'); % independent MATLAB replication
    raw=filter(1,[1 -1.6 .95],randn(T+burn,1));
    x=raw(burn+1:end); x=x/std(x,1);
    clean=filter(h,1,x).*(1-2*states(:));
    d=clean+sqrt(.001)*randn(T,1);
    v=one_run(x,d,[h -h],states,qs,etas,leakmode,cuts);
    curves=curves+v/R;
    perRun(r,:,:)=mean(v(:,31:end,:),2);
    if mod(r,20)==0, fprintf('%d/%d realizations\n',r,R); end
end

NMSD_dB=10*log10(mean(perRun(:,:,1),1)).';
EMSE_dB=10*log10(mean(perRun(:,:,2),1)).';
results=table(names.',qs.',etas.',NMSD_dB,EMSE_dB,...
    'VariableNames',{'Algorithm','q','eta','NMSD_dB','EMSE_dB'});
disp(results);
writetable(results,fullfile(root,'matlab_replica.csv'));
save(fullfile(root,'matlab_replica.mat'),'results','perRun','curves','cfg');

% --- START OF MODIFIED PLOTTING SECTION ---
figure('Color', 'w'); 
hold on;
colors = lines(2); 
names = ["BLMS", "q-BLMS"];
for c = 1:2 
    plot(cuts(4:end), 10*log10(curves(c,3:end,2)), ...
        'LineWidth', 1.5, ...
        'Color', colors(c,:), ...
        'DisplayName', names(c));
end
xlabel('Sample', 'FontWeight', 'bold');
ylabel('Clean-output prediction MSE (dB)', 'FontWeight', 'bold');
legend('Location', 'best', 'EdgeColor', 'none', 'Color', 'w', 'TextColor', 'k');
grid on;
set(gca, 'Color', 'w', ...
         'XColor', [0.15 0.15 0.15], ...
         'YColor', [0.15 0.15 0.15], ...
         'GridLineStyle', '--', ...
         'GridAlpha', 0.3, ...
         'FontSize', 11, ...
         'Box', 'on');
exportgraphics(gcf, fullfile(root, 'matlab_replica.png'), 'Resolution', 300);
% --- END OF MODIFIED PLOTTING SECTION ---
end

function metrics=one_run(x,d,hstates,states,qs,etas,leakmode,cuts)
M=size(hstates,1); K=5; T=numel(x); C=numel(qs); B=numel(cuts)-1;
W=zeros(M,C);X=zeros(M,K);dn=zeros(K,1);metrics=zeros(C,B,2);
qs=reshape(qs,1,[]);etas=reshape(etas,1,[]);
for n=1:T
    xtdl=zeros(M,1);len=min(M,n);xtdl(1:len)=x(n:-1:n-len+1);
    X=[xtdl X(:,1:end-1)]; dn=[d(n);dn(1:end-1)];
    if n<M+K-1,continue;end
    h=hstates(:,states(n)+1);
    truth=h.'*xtdl;
    pred=xtdl.'*W;
    emse=(truth-pred).^2;
    E=dn-X.'*W;
    energy=repmat(sum(X.^2,2),1,C);
    energy(:,leakmode==1)=K; % E[diag(XX')] for unit-power WSS input
    W=W+2*(X*E).*etas-energy.*W.*((qs-1).*etas);
    nmsd=sum((h-W).^2,1)/sum(h.^2);
    b=find(cuts<=n-1,1,'last');
    if b<=B
        denom=cuts(b+1)-max(cuts(b),M+K-2);
        metrics(:,b,1)=metrics(:,b,1)+nmsd.'/denom;
        metrics(:,b,2)=metrics(:,b,2)+emse.'/denom;
    end
end
end
