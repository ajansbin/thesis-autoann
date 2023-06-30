import torch


class Transformation:
    def transform(self):
        raise NotImplementedError("Calling method for abstract class Transformations")

    def untransform(self, x):
        return x


class ToTensor(Transformation):
    def transform(self, x) -> torch.tensor:
        return torch.tensor(x, dtype=torch.float32)


class Normalize(Transformation):
    def __init__(self, means: list, stds: list):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

        self.start_index = 0
        self.end_index = -1

    def transform(self, x: torch.tensor) -> torch.tensor:
        if x.ndim > 1:  # temporal encoding and score not normalized
            x[self.start_index : self.end_index] = (
                x[self.start_index : self.end_index] - self.means
            ) / self.stds
        else:
            x[0:-1] = (x[0:-1] - self.means) / self.stds
        return x

    def untransform(self, x: torch.tensor) -> torch.tensor:
        if x.ndim > 1:  # temporal encoding and score not normalized
            x[self.start_index : self.end_index] = (
                x[self.start_index : self.end_index] * self.stds
            ) + self.means
        else:
            x[0:-1] = x[0:-1] * self.stds + self.means
        return x

    def set_start_and_end_index(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index


class CenterOffset(Transformation):
    def __init__(self):
        self.offset = None

        self.start_index = 0
        self.end_index = -1

    def transform(self, x: torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
            x[self.start_index : self.end_index, 0:3] = (
                x[self.start_index : self.end_index, 0:3] - self.offset
            )
        else:
            x[0:3] = x[0:3] - self.offset
        return x

    def untransform(self, x: torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
            x[self.start_index : self.end_index, 0:3] = (
                x[self.start_index : self.end_index, 0:3] + self.offset
            )
        else:
            x[0:3] = x[0:3] + self.offset
        return x

    def set_offset(self, x: list):
        if torch.is_tensor(x):
            self.offset = x[0:3].clone().detach()
        else:
            self.offset = torch.tensor(x[0:3], dtype=torch.float32)

    def set_start_and_end_index(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index


class YawOffset(Transformation):
    def __init__(self):
        self.offset = None

        self.start_index = 0
        self.end_index = -1

    def transform(self, x: torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
            x[self.start_index : self.end_index, 6] = (
                x[self.start_index : self.end_index, 6] - self.offset
            )
        else:
            x[6] = x[6] - self.offset
        return x

    def untransform(self, x: torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
            x[self.start_index : self.end_index, 6] = (
                x[self.start_index : self.end_index, 6] + self.offset
            )
        else:
            x[6] = x[6] + self.offset
        return x

    def set_offset(self, x: list):
        if torch.is_tensor(x):
            self.offset = x[6].clone().detach()
        else:
            self.offset = torch.tensor(x[6], dtype=torch.float32)

    def set_start_and_end_index(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index


class PointsShift(Transformation):
    def __init__(self, max_shift_size):
        self.max_shift_size = max_shift_size
        self.start_index = 0
        self.end_index = -1

    def transform(self, points: torch.tensor) -> torch.tensor:
        """
        input: points   tensor (W,1000,4)
        """
        # randomly select shifting amount
        N = self.max_shift_size
        s = 2 * N * torch.rand(3) - N

        # only shift coordinates 0:3
        shifted_points = points.clone()
        shifted_points[self.start_index : self.end_index, :, 0:3] += s

        return shifted_points

    def set_start_and_end_index(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index


class PointsScale(Transformation):
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def transform(self, points: torch.tensor) -> torch.tensor:
        """
        input: points tensor (W, 1000, 4)
        """
        # randomly select scaling factor
        scaling_factor = (self.max_scale - self.min_scale) * torch.rand(
            3
        ) + self.min_scale

        # only scale coordinates 0:3
        scaled_points = points.clone()
        scaled_points[:, :, 0:3] *= scaling_factor

        return scaled_points
