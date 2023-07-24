"""
@author: Jianxiong Ye
"""

import yaml
import contextlib
from pathlib import Path
from copy import deepcopy

from models.encoder import *
from models.utils.model import model_info, fuse_conv_bn, initialize_weights
from models.utils.helper import logger, get_nearest_divisible, colorstr


module_register = [Conv, SPPF, CARAFE, RepConv]


class InferModel(nn.Module):
    def __init__(self, cfg, ch, nc):
        super().__init__()
        self.yaml_file = Path(cfg).name
        with open(cfg, encoding='ascii', errors='ignore') as f:
            self.yaml = yaml.safe_load(f)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], nc=nc)
        self.names = [str(i) for i in range(nc)]

        m = self.model[-1]
        s = 256
        m.inplace = True
        forward = lambda x: self.forward(x)[0]
        m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
        self.stride = m.stride
        m.bias_init()

        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x):
        y, dt, GFLOPs, params = [], [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, RepConv):
                m.fuse_convs()
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect
        m.stride = fn(m.stride)
        m.anchors = fn(m.anchors)
        m.strides = fn(m.strides)
        return self

    def info(self):  # print model information
        model_info(self)


Model = InferModel


def parse_model(d, ch, nc):  # model_dict, input_channels(3)
    gd, gw, act = d['depth_multiple'], d['width_multiple'], d.get('activation')
    Conv.default_act = eval(f'nn.{act}()')  # redefine default activation
    logger.info(f"{colorstr('activation:')} {act}")  # print
    logger.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['Encoder'] + d['Decoder']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in module_register:
            c1, c2 = ch[f], args[0]
            c2 = get_nearest_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in [Detector]:
            args.append([ch[x] for x in f])
            args = [nc] + args
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

