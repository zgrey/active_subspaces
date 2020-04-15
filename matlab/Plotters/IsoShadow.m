
function ISO = IsoShadow(Y1,Y2,Y3,F,Nlvl,lvl)
% A function for visualizing 3D level sets of a 3D ridge function
% Yi:   N by N matrix where N are the number of meshgrid points
% F:    N^3 by 1 matrix function evaluations paired with reshaped rows of Yi
% Nlvl: The number of level sets over the observed range of F
% lvl:  A 2 by 1 vector of lower and upper bounds specifying the level set range
N = size(Y1,1);

% start by partitioning the range of F
maxF = max(F);
minF = min(F);
% partition level sets
if Nlvl == 1
    lvlsets = mean(F);
else
    lvlsets = linspace(lvl(1),lvl(2),Nlvl);
end

% assign appropriate grid manually
YY(:,:,:,1) = Y1;
YY(:,:,:,2) = Y2;
YY(:,:,:,3) = Y3;

FV = cell(Nlvl,1); figure; 
for i=1:Nlvl
    fprintf('Building isosurface %i of %i...',i,Nlvl);
    FV{i} = isosurface(YY(:,:,:,1), YY(:,:,:,2), YY(:,:,:,3),reshape(F(1:N^3),N,N,N),lvlsets(i));
    Ptch = patch(FV{i},'FaceVertexCData',lvlsets(i)*ones(size(FV{i}.vertices,1),1),'FaceColor','flat'); 
    colorbar; grid on;
    Ptch.FaceAlpha = 0.15; Ptch.EdgeColor = 'none';
    clc;
end
