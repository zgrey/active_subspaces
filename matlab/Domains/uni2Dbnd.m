function P = uni2Dbnd(V,N)
% uniformly discretize ordered vertices into N points
% inputs:
%      V: a n by 2 set of ordered vertices
%      N: number of points for uniform discretization

% compute piecewise linear distance over boundary
len = cumsum([0; sqrt(sum(diff(V,1).^2,2))]);

% uniformly discretize lengths
uni = linspace(0,max(len),N);

% resample point by linear interpolation
% bin uniform samples for interpolation
Nc = histcounts(uni,len);

k = 0; P = zeros(N,2);
for i=1:length(Nc)
    t = linspace(0,1,Nc(i));
    P(k+1:Nc(i)+k,:) = t'*V(i,:) + (1 - t)'*V(i+1,:);
    k = sum(sqrt(sum(P.^2,2)) ~= 0);
end