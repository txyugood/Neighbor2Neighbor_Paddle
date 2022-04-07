import numpy as np
import paddle

class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            # std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            std = std * paddle.ones((shape[0], 1, 1, 1))
            # noise = torch.cuda.FloatTensor(shape, device=x.device)
            # torch.normal(mean=0.0,
            #              std=std,
            #              generator=get_generator(),
            #              out=noise)
            paddle.seed(0)
            noise = paddle.normal(mean=0.0, std=std)

            return x + noise
        # elif self.style == "gauss_range":
        #     min_std, max_std = self.params
        #     std = torch.rand(size=(shape[0], 1, 1, 1),
        #                      device=x.device) * (max_std - min_std) + min_std
        #     noise = torch.cuda.FloatTensor(shape, device=x.device)
        #     torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
        #     return x + noise
        # elif self.style == "poisson_fix":
        #     lam = self.params[0]
        #     lam = lam * paddle.ones((shape[0], 1, 1, 1))
        #     noised = torch.poisson(lam * x, generator=get_generator()) / lam
        #     return noised
        # elif self.style == "poisson_range":
        #     min_lam, max_lam = self.params
        #     lam = torch.rand(size=(shape[0], 1, 1, 1),
        #                      device=x.device) * (max_lam - min_lam) + min_lam
        #     noised = torch.poisson(lam * x, generator=get_generator()) / lam
        #     return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)