from .segmenter import UniLSeg
from loguru import logger


def build_segmenter(args):
    model = UniLSeg(args)
    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
    }, {
        'params': head,
    }]
    return model, param_list
