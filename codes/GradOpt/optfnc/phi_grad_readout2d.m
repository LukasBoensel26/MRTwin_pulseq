% Minimize loss: ||y-E.T*Ey||_2
% and return objective value phi, and derivative dx
%
% E - E = exp(1i*sum(gx*rampY + gy*rampY))
%
% m - magnetization/image in the spatial domain
% gx/gy temporal gradient waveforms
% rampX fixed gradient spatial form in x dir
% rampY ...

function [phi,dg,prediction,E] = phi_grad_readout2d(g,m,rampX,rampY)

g = reshape(g,[],2);

% set image size and number of time points in the readout
sz = rampX;                                                                                                                     
T = size(g,1);

nfact = sqrt(numel(sz));  % Fourier transform normalization coefficient                                                                     

% L2 norm for regularization
l2  = @(z) z.*conj(z);
dl2 = @(z) 2*z;

% vectorize ramps and the image
rampX = rampX(:).'; rampY = rampY(:).'; m = m(:);

% integrate over time to get field from the gradients                                                                 
g = cumsum(g,1);

% prewinder (to be relaxed in future)
g(:,1) = g(:,1) - T/2 - 1;

% compute the B0 by adding gradients in X/Y after multiplying them respective ramps
B0X = g(:,1) * rampX; B0Y = g(:,2) * rampY;
B0 = B0X + B0Y;

% encoding operator (ignore relaxation)
E = exp(1i*B0) / nfact;

% compute loss
prediction = (E'*E)*m;
loss = (prediction - m);
phi = sum(l2(loss));

% compute derivative with respect to temporal gradient waveforms
cmx = conj(dl2(loss)) * m.';
dgXY = (conj(E) * cmx + conj(E * cmx.')) .* E;

dg = zeros(size(g));
dg(:,1) = sum(1i*dgXY.*rampX,2);
dg(:,2) = sum(1i*dgXY.*rampY,2);

dg = cumsum(dg, 1, 'reverse');
dg = real(dg(:));

