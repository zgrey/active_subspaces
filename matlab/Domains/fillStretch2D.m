function [Y, conH, Z] = fillStretch2D(X,W,Nbnd)

% project data
Y = X*W;

%% build ordered convex hull of projected samples
DT = delaunayTriangulation(Y);
ind = convexHull(DT);

% sort convex hull vertices
conH = sortrows([atan2(Y(ind,2),Y(ind,1)),Y(ind,:)]); 
conH = conH(:,2:end);

%% uniformly discretize boundary of convex hull into Nbnd points
% close discrete ordering of points
conH = [conH; conH(1,:)];
conmsh = uni2Dbnd(conH,Nbnd);

%% build ordered zonotope vertices (use vararg to set max iter... later)
[~,Z] = zonotope_vertices(W,10);

% sort zonotope vertices
Z = sortrows([atan2(Z(:,2),Z(:,1)),Z]); 
Z = Z(:,2:end);

%% uniformly discretize boundary of zonotope into Nbnd points
% close discrete ordering of points
Z = [Z; Z(1,:)];
Zmsh = uni2Dbnd(Z,Nbnd);

% inform new "stretch" samples for improved surrogates
% compute intersection of zonotope and projected-data convex hull
DT = delaunayTriangulation([Zmsh; conmsh]); Y = incenter(DT);
inH = in2DZ(conH, Y); 
Y = Y(~inH,:);

