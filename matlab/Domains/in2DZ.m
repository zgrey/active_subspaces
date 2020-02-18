function ind = in2DZ(ZV,Y)
% Determine indices that fall within 2D zonotope
% inputs:
%       ZV: ordered vertices of 2D zonotope (or polytope)
%        Y: N by n matrix of candidate points
% output:
%      ind: row indices of points in zonotope

% determine number of samples
N = size(Y,1);
% if vertices are within machine precision, make them one vertex
ZV = ZV([true; sqrt(sum(diff(ZV).^2,2)) > eps / 2],:);
% close zonotope vertices by duplicating last vertex
ZV = [ZV; ZV(1,:)];

% build orthogonal compliment of half-spaces
% compute mid-edge points
midZV = (ZV(2:end,:) + ZV(1:end-1,:))./2;
% compute outer-normals in 2D
nmls = (ZV(2:end,:) - ZV(1:end-1,:))*[0 -1;1 0];
% make unitary
nmls = nmls./sqrt(sum(nmls.^2,2));
% prealloate logical vector of indices
ind = true(N,1);
% loop through samples to determine if point is in half-space
for i = 1:N
    % center on mid-edge
    shft = repmat(Y(i,:),size(midZV,1),1) - midZV;
    % if the inner product with the normal is positive, point is outside zonotope
    if max(sum(shft.*nmls,2)) > eps / 2
        ind(i) = false;
    end
end