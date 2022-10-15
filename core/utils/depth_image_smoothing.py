# modified from: https://github.com/intelligent-control-lab/Kinect_Smoothing/blob/master/kinect_smoothing/depth_image_smoothing.py
import math
import cv2
import numpy as np
import scipy.ndimage.filters as flt
from functools import partial


class HoleFilling_Filter(object):
    """Original Kinect depth image has many invalid pixels (black hole).

    This function helps you to fill the invalid pixel with the proper
    value.
    """

    def __init__(
        self, flag="min", radius=5, min_valid_depth=100, max_valid_depth=4000, min_valid_neighbors=1, max_radius=20
    ):
        """
        :param flag: string, specific methods for hole filling.
                'min': Fill in with the minimum valid value within the neighboring pixels
                'max': Fill in with the maximum valid value within the neighboring pixels
                'mode': Fill in with the mode of valid value within the neighboring pixels
                'mean': Fill in with the mean valid value within the neighboring pixels
                'fmi': Fast Matching Inpainting, refer to  'An Image Inpainting Technique Based on the Fast Marching Method'
                'ns': Fluid Dynamics Inpainting, refer to  'Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting'
        :param radius: float, radius of the neighboring area used for fill in a hole
        :param min_valid_depth: float,  a depth pixel is considered as invalid value, when depth < min_valid_depth
        :param max_valid_depth: float,  a depth pixel is considered as invalid value, when depth > max_valid_depth
        :param min_valid_neighbors: int, if the number of valid neighbors > min_valid_neighbors,
                        then replace the hole with the proper value calculated by these neighboring valid values.
                        if not, let radius = radius+1, recollect the neighboring valid value.
        :param max_radius: float, maximum radius for the neighboring area
        """
        self.flag = flag
        self.radius = radius
        self.valid_depth_min = min_valid_depth
        self.valid_depth_max = max_valid_depth
        self.min_valid_neighbors = min_valid_neighbors
        self.max_radius = max_radius
        if flag == "min":
            cal_fn = np.min
        elif flag == "max":
            cal_fn = np.max
        elif flag == "mean":
            cal_fn = np.mean
        elif flag == "mode":
            cal_fn = self._cal_mode
        else:
            cal_fn = None
        self.cal_fn = cal_fn
        if flag == "fmi":
            inpaint_fn = partial(cv2.inpaint, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        elif flag == "ns":
            inpaint_fn = partial(cv2.inpaint, inpaintRadius=radius, flags=cv2.INPAINT_NS)
        else:
            inpaint_fn = None
        self.inpaint_fn = inpaint_fn

        if flag not in self.all_flags:
            raise ("invalid  flags. Only support:", self.all_flags)

    def _get_neighbors(self, img, x, y, radius, img_h, img_w):
        """collect the neighboring  valid value within the range of
        (x-radius,x+radius), (y-radius,y+radius)

        :param img: numpy-array,
        :param x: int
        :param y: int
        :param radius: int
        :param img_h: int, height of image
        :param img_w: int height of image
        :return: (valid_neighbors,valid_num)
                        valid_neighbors: list, valid neigboring value
                        valid_num: int, number of valid_neighbors
        """
        valid_neighbors = []
        valid_num = 0
        for ii in range(-radius, radius, 1):
            yy = y + ii
            if yy < 0 or yy >= img_h:
                continue
            for jj in range(-radius, radius, 1):
                xx = x + jj
                if xx < 0 or xx >= img_w:
                    continue
                pixel = img[yy, xx]
                if pixel:
                    valid_neighbors.append(pixel)
                    valid_num += 1
        return valid_neighbors, valid_num

    def _get_neighbors_v2(self, img, x, y, radius, img_h, img_w):
        """# vectorized version collect the neighboring  valid value within the
        range of (x-radius,x+radius), (y-radius,y+radius)

        :param img: numpy-array,
        :param x: int, index along w
        :param y: int, index along h
        :param radius: int
        :param img_h: int, height of image
        :param img_w: int height of image
        :return: (valid_neighbors,valid_num)
                        valid_neighbors: list, valid neigboring value
                        valid_num: int, number of valid_neighbors
        """
        y_min = max(y - radius, 0)
        y_max = min(y + radius, img_h)
        x_min = max(x - radius, 0)
        x_max = min(x + radius, img_w)
        neighbors = img[y_min:y_max, x_min:x_max]
        valid_mask = neighbors > 0
        valid_num = valid_mask.sum()
        if valid_num == 0:
            valid_neighbors = []
        else:
            valid_neighbors = neighbors[valid_mask]
        return valid_neighbors, valid_num

    def _cal_mode(self, nums):
        """calculate the mode.

        :param nums: list
        :return: mode of nums
        """
        result = nums[0]
        cnt = 0
        for num in nums:
            if cnt == 0 or num == result:
                cnt += 1
                result = num
            else:
                cnt -= 1
        return result

    def statistical_smoothing(self, image):
        """smoothing image with statistical filling method, such as min,max,
        mode, mean.

        :param image: numpy-array,
        :return: smoothed: numpy-array, smoothed image
        """
        smoothed = image.copy()
        h, w = image.shape
        image[image <= self.valid_depth_min] = 0
        image[image >= self.valid_depth_max] = 0
        invalid_y, invalid_x = np.where(image == 0)

        for y, x in zip(invalid_y, invalid_x):
            _r = self.radius
            valid_neighbors, valid_num = [], 0
            while valid_num < self.min_valid_neighbors and _r < self.max_radius:
                if _r == 1:
                    valid_neighbors, valid_num = self._get_neighbors(image, x, y, _r, h, w)
                else:
                    valid_neighbors, valid_num = self._get_neighbors_v2(image, x, y, _r, h, w)
                _r += 1
            if len(valid_neighbors) > 0:
                smoothed[y, x] = self.cal_fn(valid_neighbors)

        return smoothed

    def inpainting_smoothing(self, image):
        """smoothing image with inpainting method, such as FMI, NS.

        :param image: numpy-array,
        :return: smoothed: numpy-array, smoothed image
        """
        # image[image <= self.valid_depth_min] = 0
        # image[image >= self.valid_depth_max] = 0
        # mask = np.zeros(image.shape, dtype=np.uint8)
        # mask[image == 0] = 1
        mask = (np.logical_or(image <= self.valid_depth_min, image >= self.valid_depth_max)).astype("uint8")
        smoothed = self.inpaint_fn(image, mask[:, :, np.newaxis])
        smoothed[0] = smoothed[2]  # first 2 lines is zero
        smoothed[1] = smoothed[2]
        return smoothed

    def smooth_image(self, image):
        """smooth the image using specific method.

        :param image: numpy-array,
        :return: smoothed_image: numpy-array,
        """
        image = image.copy()
        if self.flag in ["min", "max", "mean", "mode"]:
            smoothed_image = self.statistical_smoothing(image)
        elif self.flag in ["fmi", "ns"]:
            smoothed_image = self.inpainting_smoothing(image)
        else:
            raise ("invalid smoothing flags")
        return smoothed_image

    def smooth_image_frames(self, image_frames):
        """smooth image frames.

        :param image_frames: list of numpy array
        :return: smoothed_frames: list of numpy array
        """
        smoothed_frames = []
        for imgs in image_frames:
            if not isinstance(imgs, list):
                imgs = [imgs]
            for img in imgs:
                res_img = self.smooth_image(img)
                smoothed_frames.append(res_img)
        return smoothed_frames

    @property
    def all_flags(self):
        flags = [
            "min",
            "max",
            "mean",
            "mode",
            "fmi",  # Fast matching inpainting
            "ns",  # NS inpainting
        ]
        return flags


class Denoising_Filter(object):
    """Denoising filter can be used to improve the resolution of the depth
    image."""

    def __init__(
        self,
        flag="modeling",
        theta=30,
        threshold=10,
        depth_min=100,
        depth_max=4000,
        ksize=5,
        sigma=0.1,
        niter=1,
        kappa=50,
        gamma=1,
        option=1,
    ):
        """
        :param flag: string, specific methods for denoising.
                        'modeling': filter with Kinect V2 noise model,  'Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking'
                        'modeling_pf': another Kinect V2 noise modeling by Peter Fankhauser, 'Kinect v2 for Mobile Robot Navigation: Evaluation and Modeling'
                        'anisotropic': smoothing with anisotropic filtering, 'Scale-space and edge detection using anisotropic diffusion'
                        'gaussian': smoothing with Gaussian filtering
        :param theta: float, the average angle between Kinect z-axis and the object plane.
                        Used to calculate noise in the 'modeling'  and 'modeling_pf' method
        :param threshold: int, thrshold for 'modeling' and 'modeling_pf' method.
        :param depth_min: float,  minimum valid depth, we only filter the area of depth > depth_min
        :param depth_max: float,  maximum valid depth, we only filter the area of depth < depth_max
        :param ksize: int, Gaussian kernel size
        :param sigma: float, Gaussian kernel standard deviation
        :param niter: int, number of iterations for anisotropic filtering
        :param kappa: int, conduction coefficient for anisotropic filtering, 20-100 ?
        :param gamma: float, max value of .25 for stability
        :param option: 1 or 2, options for anisotropic filtering
                        1: Perona Malik diffusion equation No. 1
                2: Perona Malik diffusion equation No. 2
        """
        self.flag = flag
        self.f_x = 585  # focal length of the Kinect camera in pixel
        if flag == "modeling" or flag == "modeling_pf":
            self.threshold = threshold
            self.depth_min = depth_min
            self.depth_max = depth_max
            self.theta = theta
            self.noise_type = "pf" if flag == "modeling_pf" else "normal"
            self.filter = partial(
                self.modeling_filter,
                theta=theta,
                threshold=threshold,
                depth_min=depth_min,
                depth_max=depth_max,
                noise_type=self.noise_type,
            )
        elif flag == "gaussian":
            self.ksize = ksize
            self.sigma = sigma
            self.filter = partial(cv2.GaussianBlur, ksize=(ksize, ksize), sigmaX=0)
        elif flag == "anisotropic":
            self.niter = niter
            self.kappa = kappa
            self.gamma = gamma
            self.sigma = sigma
            self.option = option
            self.filter = partial(self.anisotropic_filter, niter=niter, kappa=kappa, gamma=gamma, sigma=sigma)

        if flag not in self.all_flags:
            raise ("invalid  flags. Only support:", self.all_flags)

    def _axial_noise(self, z, theta):
        """calculate the axial noise based on 'Modeling Kinect Sensor Noise for
        Improved 3D Reconstruction and Tracking'.

        :param z: float, depth
        :param theta: float, angle
        :return: sigma: float, axial noise
        """
        z = z / 1000
        theta = math.pi * theta / 180
        sigma = 0.0012 + 0.0019 * (z - 0.4) ** 2 + (0.0001 / np.sqrt(z)) * (theta**2 / (math.pi / 2 - theta) ** 2)
        sigma = sigma * 1000
        return sigma

    def _lateral_noise(self, z, theta):
        """calculate the lateral noise based on 'Modeling Kinect Sensor Noise
        for Improved 3D Reconstruction and Tracking'.

        :param z: float, depth
        :param theta: float, angle
        :return: sigma: float, lateral noise
        """
        f_x = 585  # 357
        theta = math.pi * theta / 180
        sigma_pixel = 0.8 + 0.035 * theta / (math.pi / 2 - theta)
        sigma = sigma_pixel * z / f_x
        return sigma

    def _axial_noise_pf(self, z, theta):
        """calculate the axial noise based on 'Kinect v2 for Mobile Robot
        Navigation: Evaluation and Modeling'.

        :param z: float, depth
        :param theta: float, angle
        :return: sigma: float, axial noise
        """
        z = z / 1000
        theta = math.pi * theta / 180
        sigma = (
            1.5
            - 0.5 * z
            + 0.3 * np.power(z, 2)
            + 0.1 * np.power(z, 3 / 2) * np.power(theta, 2) / np.power(math.pi / 2 - theta, 2)
        )
        return sigma

    def _lateral_noise_pf(self, z, theta=0, shadow=False):
        """calculate the lateral noise based on 'Kinect v2 for Mobile Robot
        Navigation: Evaluation and Modeling'.

        :param z: float, depth
        :param theta: float, angle
        :param shadow: bool,
        :return: sigma: float, lateral noise
        """
        if shadow:
            sigma = 3.1
        else:
            sigma = 1.6
        sigma = sigma * np.ones(z.shape)
        return sigma

    def anisotropic_filter(self, img, niter=1, kappa=50, gamma=0.1, step=(1.0, 1.0), sigma=0, option=1):
        """
        Anisotropic diffusion.
        usage: imgout = anisodiff(im, niter, kappa, gamma, option)

        :param img:    - input image
        :param  niter:  - number of iterations
        :param  kappa:  - conduction coefficient 20-100 ?
        :param  gamma:  - max value of .25 for stability
        :param  step:   - tuple, the distance between adjacent pixels in (y,x)
        :param  option: - 1 Perona Malik diffusion equation No 1
                       2 Perona Malik diffusion equation No 2

        :return: imgout   - diffused image.

        kappa controls conduction as a function of the gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.

        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)

        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes

        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.

        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.

        Original MATLAB code by Peter Kovesi
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>

        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>

        June 2000  original version.
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """

        # initialize output array
        img = img.astype("float32")
        imgout = img.copy()

        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()

        for ii in np.arange(1, niter):

            # calculate the diffs
            deltaS[:-1, :] = np.diff(imgout, axis=0)
            deltaE[:, :-1] = np.diff(imgout, axis=1)

            if 0 < sigma:
                deltaSf = flt.gaussian_filter(deltaS, sigma)
                deltaEf = flt.gaussian_filter(deltaE, sigma)
            else:
                deltaSf = deltaS
                deltaEf = deltaE

            # conduction gradients (only need to compute one per dim!)
            if option == 1:
                gS = np.exp(-((deltaSf / kappa) ** 2.0)) / step[0]
                gE = np.exp(-((deltaEf / kappa) ** 2.0)) / step[1]
            elif option == 2:
                gS = 1.0 / (1.0 + (deltaSf / kappa) ** 2.0) / step[0]
                gE = 1.0 / (1.0 + (deltaEf / kappa) ** 2.0) / step[1]

            # update matrices
            E = gE * deltaE
            S = gS * deltaS

            # subtract a copy that has been shifted 'North/West' by one
            # pixel. don't as questions. just do it. trust me.
            NS[:] = S
            EW[:] = E
            NS[1:, :] -= S[:-1, :]
            EW[:, 1:] -= E[:, :-1]

            # update the image
            imgout += gamma * (NS + EW)

        return imgout

    def modeling_filter(self, img, theta=30, threshold=10, depth_min=100, depth_max=4000, noise_type="normal"):
        """modeling the noise distribution and filtering based on noise model.

        :param img: numpy-array,
        :param theta: float, average angle between kinect z-axis and object plane.
        :param threshold: int, thrshold for 'modeling' and 'modeling_pf' method.
        :param depth_min: float,  minimum valid depth, we only filter the area of depth > depth_min
        :param depth_max: float,  maximum valid depth, we only filter the area of depth < depth_max
        :param noise_type: 'normal' of 'pf',
                        'normal': noise modeling based on 'Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking'
                        'pf': noise modeling based on 'Kinect v2 for Mobile Robot Navigation: Evaluation and Modeling'
        :return: denoised img: numpy-array
        """

        denoised_img = img.copy()
        h, w = img.shape
        lateral_noise = self._lateral_noise_pf if noise_type == "pf" else self._lateral_noise
        axial_noise = self._axial_noise_pf if noise_type == "pf" else self._axial_noise
        l_noise = np.power(lateral_noise(img, theta), 2)
        z_noise = np.power(axial_noise(img, theta), 2)
        distance_metrics = np.array([[1.414, 1, 1.414], [1, 0, 1], [1.414, 1, 1.414]])
        for x in range(h):
            for y in range(w):
                D_u = img[x, y]
                if D_u >= depth_min and D_u <= depth_max:
                    sigmal_l = l_noise[x, y]
                    sigmal_z = z_noise[x, y]
                    D_k = img[max(x - 1, 0) : x + 2, max(y - 1, 0) : y + 2]
                    delta_z = abs(D_k - D_u)
                    delta_u = distance_metrics[int(x == 0) : 2 + int(x < h - 1), int(y == 0) : 2 + int(y < w - 1)]
                    mark = delta_z < threshold
                    D_k_list = D_k[mark].flatten()
                    u_list = delta_u[mark].flatten()
                    z_list = delta_z[mark].flatten()
                    if len(D_k_list) > 0:
                        w_k_list = np.exp(-(u_list**2) / (2 * sigmal_l) - z_list**2 / (2 * sigmal_z))
                        denoised_img[x, y] = np.sum(D_k_list * w_k_list) / w_k_list.sum()
        return denoised_img

    def smooth_image(self, image):
        """smooth the image using a specified method.

        :param image: numpy-array,
        :return: smoothed_image: numpy-array,
        """
        image = image.copy()
        smoothed = self.filter(image)
        return smoothed

    def smooth_image_frames(self, image_frames):
        """smooth image frames.

        :param image_frames: list of numpy array
        :return: smoothed_frames: list of numpy array
        """
        smoothed_frames = []
        for imgs in image_frames:
            if not isinstance(imgs, list):
                imgs = [imgs]
            for img in imgs:
                res_img = self.smooth_image(img)
                smoothed_frames.append(res_img)
        return smoothed_frames

    @property
    def all_flags(self):
        flags = [
            "modeling",
            "modeling_pf",
            "gaussian",
            "anisotropic",
        ]
        return flags


if __name__ == "__main__":
    from lib.vis_utils.image import heatmap, grid_show
    import mmcv
    import cv2
    import time
    from tqdm import tqdm

    depth = (mmcv.imread("datasets/BOP_DATASETS/ycbv/test/000048/depth/000001.png", "unchanged") / 10000.0).astype(
        "float32"
    )
    hole_fill = HoleFilling_Filter(
        flag="mean", radius=2, min_valid_depth=0.1, max_valid_depth=4, min_valid_neighbors=1, max_radius=3
    )
    print("start hole filling...")
    run = 1
    total_time = 0
    for i in tqdm(range(run)):
        tic = time.time()
        depth_aug = hole_fill.smooth_image(depth)
        total_time += time.time() - tic
    print("{}s/img, {}fps".format(total_time / run, run / total_time))
    diff = depth_aug - depth
    diff_abs = np.abs(diff)
    print("diff min: {} max: {}, mean: {}".format(diff_abs.min(), diff_abs.max(), diff_abs.mean()))
    grid_show(
        [heatmap(depth, to_rgb=True), heatmap(depth_aug, to_rgb=True), heatmap(diff, to_rgb=True)],
        ["depth", "depth_aug", "diff"],
        row=1,
        col=3,
    )
