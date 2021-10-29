import torch


class ToHeatmap(torch.nn.Module):
    def __init__(self, radius=5):
        super().__init__()

        bounds = torch.arange(-radius, radius+1, 1.0)
        y, x = torch.meshgrid(bounds, bounds)
        kernel = (-(x ** 2 + y ** 2) / (2 * radius ** 2)).exp()
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

        self.r = radius
        self.register_buffer('kernel', kernel)

    def forward(self, points, img):
        n, _, h, w = img.shape
        heatmap = torch.zeros((n, h, w)).type_as(img)

        for i in range(n):
            output = heatmap[i]

            cx, cy = points[i].round().long()
            cx = torch.clamp(cx, 0, w-1)
            cy = torch.clamp(cy, 0, h-1)

            left = min(cx, self.r)
            right = min(w - 1 - cx, self.r)
            bot = min(cy, self.r)
            top = min(h - 1 - cy, self.r)

            output_crop = output[cy-bot:cy+top+1, cx-left:cx+right+1]
            kernel_crop = self.kernel[self.r-bot:self.r+top+1, self.r-left:self.r+right+1]
            output_crop[...] = kernel_crop

        return heatmap


if __name__ == '__main__':
    h = 64
    w = 128
    n = 8

    torch.manual_seed(0)

    layer = ToHeatmap()
    img = torch.randn(n, 1, h, w)
    points = torch.clamp(torch.randn(n, 2), -1.25, 1.25)
    points[:, 0] *= w
    points[:, 1] *= h

    heatmap = layer(points, img)

    import matplotlib.pyplot as plt

    for i in range(n):
        plt.title('%.3f %.3f' % tuple(points[i]))
        plt.imshow(heatmap[i])
        plt.show()
