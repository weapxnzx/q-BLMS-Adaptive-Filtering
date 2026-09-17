clear all;
clc;
close all;
%% =====================================================================
%  GENERAL PARAMETERS
%  =====================================================================
itn      = 6000;
sigman2  = 0.001;
sigman   = sqrt(sigman2);
wo       = fir1(120-1, 0.4);
wo       = wo(:);
runs     = 200;              % increased from 50 -> 200 for better averaging
N        = 120;
M        = 5;
n_jump   = 3000;              % system change instant
% Window used to measure steady-state "floor"
% (last samples before the jump)
win_floor = 500;
% Threshold (dB) to measure reconvergence time after the jump
threshold_reconv_dB = -10;      % adjust according to your curves' range
% ----------------------------------------------------
% q VALUES TO EVALUATE (extended sweep)
q_values = [1.0, 1.02, 1.1, 1.3];
num_q    = length(q_values);
% ----------------------------------------------------
% PARAMETERS FOR THE ILL-CONDITIONED INPUT GENERATOR FILTER
% Complex poles close to the unit circle -> narrow spectral peak
% -> autocorrelation matrix with large eigenvalue spread.
b_color = 1;
a_color = [1 -1.6 0.95];
%% =====================================================================
%  ACCUMULATOR INITIALIZATION
%  =====================================================================
xi1      = zeros(itn,1);
mu1      = 0.001;
xi_b     = zeros(itn,1);
mu_b     = 0.0005;
xi_q     = zeros(num_q, itn);
miss_q   = zeros(num_q, itn);
mu_q     = 0.0005;
miss1_avg = zeros(itn,1);
miss_b_avg = zeros(itn,1);
%% =====================================================================
%  MONTE CARLO LOOP
%  =====================================================================
for k = 1:runs
    % ---- Ill-conditioned input generation (narrowband) ----
    x_raw = randn(itn,1);
    x = filter(b_color, a_color, x_raw);
    x = x / std(x);            % normalize power to 1
    d1 = filter(wo,1,x);
    d  = d1 + (sigman*randn(itn,1));
    w1 = zeros(N,1);
    wb = zeros(N,1);
    Xb = zeros(N,M);
    wq = zeros(N, num_q);
    Xq = zeros(N, M, num_q);
    % Precalculate q_vec for each q (constant, outside the n loop)
    q_vecs = q_values .* ones(N, num_q);   % N x num_q
    for n = N+M-1:itn
        if n == n_jump
            wo = wo*(-1);
            d1 = filter(wo,1,x);
            d  = d1 + (sigman*randn(itn,1));
        end
        xtdl = x(n:-1:n-N+1);
        % ---------------- LMS Filter ----------------
        e1 = d(n) - w1'*xtdl;
        w1 = w1 + 2*mu1*e1*xtdl;
        xi1(n) = xi1(n) + e1^2;
        miss1_avg(n) = miss1_avg(n) + (norm(wo-w1))/(norm(wo));
        % ---------------- Standard BLMS Filter ----------------
        Xb = [xtdl Xb(:,1:M-1)];
        Yhat_b = Xb' * wb;
        Eb = d(n:-1:n-M+1) - Yhat_b;
        wb = wb + 2 * mu_b * (Xb * Eb);
        xi_b(n) = xi_b(n) + Eb(1)^2;
        miss_b_avg(n) = miss_b_avg(n) + (norm(wo-wb))/(norm(wo));
        % ---------------- q-BLMS Filter ----------------
        % The input block Xq_curr DOES NOT depend on q, it is calculated
        % only once and reused for all q values.
        Xq_curr = [xtdl Xq(:,1:M-1,1)];   % same structure for all
        XXT_diag = sum(Xq_curr.^2, 2);
        for i_q = 1:num_q
            Xq(:,:,i_q) = Xq_curr;
            wq_curr = wq(:, i_q);
            q_vec = q_vecs(:, i_q);
            Yhat_q = Xq_curr' * wq_curr;
            Eq = d(n:-1:n-M+1) - Yhat_q;
            e_q = Eq(1);
            q_correction = (q_vec - 1) .* wq_curr .* XXT_diag;
            wq_curr = wq_curr + 2 * mu_q * (Xq_curr * Eq) - mu_q * q_correction;
            wq(:, i_q) = wq_curr;
            xi_q(i_q, n) = xi_q(i_q, n) + e_q^2;
            miss_q(i_q, n) = miss_q(i_q, n) + (norm(wo-wq_curr))/(norm(wo));
        end
    end
end
%% =====================================================================
%  AVERAGES
%  =====================================================================
xi1  = xi1 / runs;
xi_b = xi_b / runs;
xi_q = xi_q / runs;
miss1_avg  = miss1_avg / runs;
miss_b_avg = miss_b_avg / runs;
miss_q     = miss_q / runs;
miss1     = 20*log10(miss1_avg);
miss_b    = 20*log10(miss_b_avg);
miss_q_db = 20*log10(miss_q);
%% =====================================================================
%  PLOTS
%  =====================================================================
colors = lines(num_q + 2);
% ---- Figure 1: MSE ----
figure(1);
set(gcf, 'Color', 'w'); 
hold on;
plot(1:itn, 10*log10(xi1), 'DisplayName', 'LMS');
for i_q = 1:num_q
    plot(1:itn, 10*log10(xi_q(i_q,:)), '-', 'Color', colors(i_q+1,:), ...
        'DisplayName', sprintf('q-BLMS (q=%.2f)', q_values(i_q)));
end
plot(1:itn, 10*log10(xi_b), 'DisplayName', 'BLMS');
grid on;
set(gca, 'Color', 'w', ...
         'XColor', 'k', ...
         'YColor', 'k', ...
         'GridLineStyle', '--', ...
         'GridAlpha', 0.3);
xlabel('Number of iterations');
ylabel('MSE (dB)');
title('MSE - Ill-conditioned input', 'Color', 'k'); 
legend('Color', 'w', 'TextColor', 'k', 'Location', 'best'); 
hold off;
% ---- Figure 2: Misalignment ----
figure(2);
hold on;
set(gcf, 'Color', 'w'); 
plot(1:itn, miss1, ':', 'DisplayName', 'LMS');
plot(1:itn, miss_b, '-.', 'DisplayName', 'BLMS', 'LineWidth', 1.5);
for i_q = 1:num_q
    plot(1:itn, miss_q_db(i_q,:), '-', 'Color', colors(i_q+2,:), ...
        'DisplayName', sprintf('q-BLMS (q=%.2f)', q_values(i_q)), 'LineWidth', 1.2);
end
xline(n_jump, '--k', 'HandleVisibility','off');
grid on;
set(gca, 'Color', 'w', ...
         'XColor', 'k', ...
         'YColor', 'k', ...
         'GridLineStyle', '--', ...
         'GridAlpha', 0.3);
xlabel('Number of iterations');
ylabel('Misalignment (dB)');
title('Misalignment - Ill-conditioned input', 'Color', 'k');
legend('Color', 'w', 'TextColor', 'k', 'Location', 'best'); 
hold off;
%% =====================================================================
%  AUXILIARY FUNCTION
%  =====================================================================
function t_reconv = calculate_reconvergence_time(curve_dB, n_jump, itn, threshold, verif_win)
% Finds the first instant n > n_jump where the curve crosses below
% the threshold and stays below for 'verif_win' consecutive samples.
% Returns the number of iterations elapsed since the jump (n - n_jump).
    t_reconv = NaN;
    for n = n_jump : (itn - verif_win)
        window = curve_dB(n:n+verif_win-1);
        if all(window < threshold)
            t_reconv = n - n_jump;
            break;
        end
    end
    if isnan(t_reconv)
        t_reconv = itn - n_jump;  % did not reconverge within the simulated horizon
    end
end

%% =====================================================================
%  SAVE FIGURES
%  =====================================================================
% Create 'results' directory if it doesn't exist
if ~exist('results', 'dir')
    mkdir('results');
end

% Save Figure 1 (MSE)
figure(1);
exportgraphics(gcf, fullfile('results', 'mse_ill_conditioned.png'), 'Resolution', 300);
fprintf('Saved MSE plot to results/mse_ill_conditioned.png\n');

% Save Figure 2 (Misalignment)
figure(2);
exportgraphics(gcf, fullfile('results', 'misalignment_ill_conditioned.png'), 'Resolution', 300);
fprintf('Saved Misalignment plot to results/misalignment_ill_conditioned.png\n');