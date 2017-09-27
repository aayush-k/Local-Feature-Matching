% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays


% 1. Compute the horizontal and vertical derivatives of the image Ix and Iy by convolving the original image with derivatives of Gaussians (Section 3.2.3).
% 2. Compute the three images corresponding to the outer products of these gradients. (The matrix A is symmetric, so only three entries are needed.)
% 3. Convolve each of these images with a larger Gaussian.
% 4. Compute a scalar interest measure using one of the formulas discussed above.
% 5. Find local maxima above a certain threshold and report them as detected feature point locations.

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or (b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% BLUR image??

% alpha = 0.06 as proposed by harris
alpha = 0.04;
threshold = 0.02

% gaussian kernels for the original image and the derivatives
smallGaussian = fspecial('gaussian', [feature_width, feature_width], 1);
largeGaussian = fspecial('gaussian', [feature_width, feature_width], 2);
% filter original image
GausImg = imfilter(image, smallGaussian);
% calculate derivative products
[I_x, I_y] = imgradientxy(GausImg);
I_x2 = I_x .^ 2;
I_y2 = I_y .^ 2;
I_xy = I_x .* I_y;
% apply gaussian to derivative products
GausI_x2 = imfilter(I_x2, largeGaussian);
GausI_y2 = imfilter(I_y2, largeGaussian);
GausI_xy = imfilter(I_xy, largeGaussian);
% calculate har
har = (GausI_x2 .* GausI_y2) - (GausI_xy .^ 2) - (alpha .* (GausI_x2 + GausI_y2) .^ 2);

% remove edges to account for wrong detections
har(1:feature_width, :) = 0;
har(:, 1:feature_width) = 0;
har(end - feature_width:end, :) = 0;
har(:, end - feature_width:end) = 0;

% threshold the har and extract connected componented
threshedHar = har > threshold;
CC = bwconncomp(threshedHar);

% get dimensions
[rows, cols] = size(har);

% initialize arrays to 
x = zeros(CC.NumObjects, 1);
y = zeros(CC.NumObjects, 1);
confidence = zeros(CC.NumObjects, 1);

for ind = 1:CC.NumObjects
    % get indices of blob regions
    blobPixelInds = CC.PixelIdxList{ind}
    % mask to get original har values
    blobPixelVals = har(blobPixelInds)
    % get maximum
    [maxV, maxI] = max(blobPixelVals)

    % column indices
    x(ind) = floor(blobPixelInds(maxI) / rows);
    y(ind) = mod(blobPixelInds(maxI), rows);
    confidence(ind) = maxV;

end



% BWLABEL (old), BWCONNCOMP (new)
%   take max value in each component

%

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in
% thresholded binary image. You could, for instance, take the maximum value
% within each component.

%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Placeholder that you can delete -- random points
% x = ceil(rand(500,1) * size(image,2));
% y = ceil(rand(500,1) * size(image,1));

end

