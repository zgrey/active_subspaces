function [sub,surr1D,surr2D,fig] = FD_model(X,F,dF,deg,Nnew,Nz)
%% Global quadratic model (reduces to arbitrary r-dimensional subspace).
% inputs:
%      X: a N by m matrix of parameter values from the [-1,1]^m hypercube
%      F: a N by 1 vector of paired function evaluations to the rows of X
%    deg: an integer specifying the degree of the monomial surrogate
%  100: an integer specifying the number of uniform samples of the surrogate
%
% outputs:
%    sub: a structure array containing the subspace information
% surr1D: a surrogate structure for the 1-dim. surrogate
% surr2D: a surrogate structure for the 2-dim. surrogate
%    fig: the figure structure array for the plots
%        -> fig.shdw1D: the figure object for the 1-dim. shadow plots
%        -> fig.shdw2D: the figure object for the 2-dim. shadow plots
%        -> fig.eigvals: the figure object for the eigenvalues
%        -> fig.eigvecs: the figure object for the eigenvectors

%% Active subspace
% Approximate the active subspace by a global quadratic model
% the number of random samples and parameters
N = size(X,1); m = size(X,2);
[sub.W,sub.eigs] = svd(1/sqrt(N)*dF',0);
% take the magnitude (in case of noise)
sub.eigs = abs(sub.eigs);
% ensure sort in descending order
[sub.eigs, ind] = sort(diag(sub.eigs), 1, 'descend');
% index corresponding eigenvectors
sub.W = sub.W(:, ind);

%% Domain
% 1D domain
% find the extents of the 1-dimensional active variable domain
% uniformly sample over extent of projected samples
sub.Y1D = X*sub.W(:,1); surr1D.dom.Y = linspace(min(sub.Y1D),max(sub.Y1D),100)';
% determine the extent of the original domain projected to the new 1d-domain
options = optimset('Display','off');
ext1 = linprog(sub.W(:,1),[],[],[],[],-ones(m,1),ones(m,1),options);
ext2 = linprog(-sub.W(:,1),[],[],[],[],-ones(m,1),ones(m,1),options);
% assign the extent of the 1-dim. domain
surr1D.dom.ext = [min([ext1'; ext2']*sub.W(:,1)), max([ext1'; ext2']*sub.W(:,1))];
% uniformly sample over the extent of 1-dim. domain for visualization
surr1D.dom.uniViz = linspace(min(surr1D.dom.ext),max(surr1D.dom.ext),100)';
% compute active coordinates for stretch sampling
surr1D.dom.newY = [linspace(surr1D.dom.ext(1),min(sub.Y1D),ceil(Nnew/2)),...
                   linspace(max(sub.Y1D),surr1D.dom.ext(2),ceil(Nnew/2))]';
% preallocate
surr1D.dom.newX = zeros(Nz*size(surr1D.dom.newY,1),m);
% compute inactive samples
for i = 1:size(surr1D.dom.newY,1)
    % 1D domain
    surr1D.dom.Z = hit_and_run_z(Nz,surr1D.dom.newY(i),sub.W(:,1),sub.W(:,2:end));
    surr1D.dom.newX(Nz*(i-1) + 1:Nz*(i-1) + Nz,:) = repmat(surr1D.dom.newY(i),Nz,1)*sub.W(:,1)' + surr1D.dom.Z*sub.W(:,2:end)';
end

% 2D domain
[surr2D.dom.newY, sub.conH, sub.Z] = fillStretch2D(X,sub.W(:,1:2),Nnew);
% define the extent of the surrogate domain to be the zonotope
surr2D.dom.ext = sub.Z;
% sample uniform grid of min and max extent of domain zonotope for visualization
[Y1d2,Y2d2] = meshgrid(linspace(min(surr2D.dom.ext(:,1)),max(surr2D.dom.ext(:,1)),100)',...
                       linspace(min(surr2D.dom.ext(:,2)),max(surr2D.dom.ext(:,2)),100)');
surr2D.dom.uniViz = [reshape(Y1d2,100^2,1), reshape(Y2d2,100^2,1)];

% preallocate
surr2D.dom.newX = zeros(Nz*size(surr2D.dom.newY,1),m);
% compute inactive samples for stretch sampling
for i = 1:size(surr2D.dom.newY,1)
    % 2D domain 
    surr2D.dom.Z = hit_and_run_z(Nz,surr2D.dom.newY(i,:)',sub.W(:,1:2),sub.W(:,3:end));
    surr2D.dom.newX(Nz*(i-1) + 1:Nz*(i-1) + Nz,:) = repmat(surr2D.dom.newY(i,:),Nz,1)*sub.W(:,1:2)' + surr2D.dom.Z*sub.W(:,3:end)';
end

%% Surrogate models
% 1-dimensional surrogate models
% train a 1-dim. polynomial surrogate
[surr1D.Coef, ~, ~, ~, surr1D.Hx, surr1D.res] = poly_train(X*sub.W(:,1), F, deg);
surr1D.Rsqd = 1 - cov(surr1D.res)./cov(F);
% obtain the surrogate response over the full domain
[surr1D.H, surr1D.dH] = poly_predict(surr1D.dom.uniViz,surr1D.Coef,deg);
[surr1D.Hy, surr1D.dHy] = poly_predict(surr1D.dom.Y,surr1D.Coef,deg);

% 2-dimensional surrogate models
% train a 2-dim. polynomial surrogate
[surr2D.Coef, ~, ~, ~, surr2D.Hx, surr2D.res] = poly_train(X*sub.W(:,1:2), F, deg);
surr2D.Rsqd = 1 - cov(surr2D.res)./cov(F);
% obtain the surrogate response over the full domain
[surr2D.H, surr2D.dH] = poly_predict(surr2D.dom.uniViz,surr2D.Coef,deg);
% remove extrapolations outside zonotope for visualizatoin
surr2D.H(~in2DZ(surr2D.dom.ext,surr2D.dom.uniViz)) = NaN;

%% Visualize
% make 1-dim. shadow plots
fig.shdw1D = figure;
% make a 1-dimensional shadow plot
scatter(X*sub.W(:,1),F,50,'filled'); alpha(0.5); hold on;
% plot the resulting approximation
plot(surr1D.dom.Y,surr1D.Hy,'k','linewidth',2);
plot(surr1D.dom.uniViz,surr1D.H,'k--','linewidth',2);
ax = gca; Ylim = ax.YLim; ax.YLim(1) = Ylim(1);
% plot the extent of the original domain
scatter(surr1D.dom.ext,min(ax.YLim)*ones(1,2),50,'filled','k');
% plot the extent of the projected sample domain
scatter([min(sub.Y1D), max(sub.Y1D)],min(ax.YLim)*ones(1,2),50,'k');
% plot the points for resampling
scatter(surr1D.dom.newY,min(ax.YLim)*ones(1,2*ceil(Nnew/2)),50,'k*');
title(['1D Shadow Plot - FD Model, R^2 = ', num2str(surr1D.Rsqd)]);
xlabel 'y = w_1^Tx'; ylabel 'f';
legend('Original smpls',['order-',num2str(deg), ' poly.'])

% make 2-dim. shadow plots
fig.shdw2D = figure;
% plot the projected samples (shadow plot)
scatter(X*sub.W(:,1),X*sub.W(:,2),50,'filled','cdata',F);
alpha(0.8); hold on; axis square; lim = caxis;
% plot the surrogate over the extent of the original domain
contour(reshape(surr2D.dom.uniViz(:,1),100,100),reshape(surr2D.dom.uniViz(:,2),100,100)...
        ,reshape(surr2D.H,100,100),50,'linewidth',1); hold on;
caxis(lim);
% plot extent of 2-dim. domain vertices
scatter(surr2D.dom.ext(:,1),surr2D.dom.ext(:,2),50,'k','filled');
% plot extent of projected data
scatter(sub.conH(:,1),sub.conH(:,2),50,'k');
% plot the edges of the domain zonotope vertices
plot([surr2D.dom.ext(:,1); surr2D.dom.ext(1,1)],...
     [surr2D.dom.ext(:,2); surr2D.dom.ext(1,2)],'k','linewidth',2);
% plot the edges of the projected data convex hull
plot([sub.conH(:,1); sub.conH(1,1)],...
     [sub.conH(:,2); sub.conH(1,2)],'k--','linewidth',1);
% plot new active variable samples
scatter(surr2D.dom.newY(:,1),surr2D.dom.newY(:,2),55,'ko');
title(['2D Shadow Plot - FD Model, R^2 = ' num2str(surr2D.Rsqd)]); colorbar;
xlabel 'y_1 = w_1^Tx'; ylabel 'y_2 = w_2^Tx';

% plot subspace values
fig.eigvals = figure;
% plot eigenvalues
stem(sub.eigs,'filled'); grid on; ax = gca; ax.YScale = 'log';
title 'Eigenvalues - FD Model';
fig.eigvecs = figure;
% plot eigenvectors
stem(sub.W(:,1:2),'filled'); grid on; hold on;
title 'Eigenvectors - FD Model'; legend('1st eig. vec.','2nd eig. vec.')