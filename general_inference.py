import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger
from typing import List, Union

import utils.config as config
from model import build_segmenter
from utils.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def convert(img):
    mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258,
                                0.27577711]).reshape(3, 1, 1)
    # Image ToTensor & Normalize
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    if not isinstance(img, torch.FloatTensor):
        img = img.float()
    img.div_(255.).sub_(mean).div_(std)
    return img


def getTransformMat(img_size, input_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


def run_demo(args, model, img, expression, save_path, ori_img, params):
    model.eval()
    # data
    img = img.cuda(non_blocking=True)
    sent = expression
    text = tokenize(sent, args.word_len, True)
    text = text.cuda(non_blocking=True)
    # inference
    img = img.unsqueeze(0)

    
    pred = model(img, text)
    if pred.shape[-2:] != img.shape[-2:]:
        pred = F.interpolate(pred,
                            size=img.shape[-2:],
                            mode='bilinear',
                            align_corners=True).squeeze()
    pred = torch.sigmoid(pred)
    h, w = params['ori_size']
    mat = params['inverse']

    pred = pred.cpu().numpy()
    pred = cv2.warpAffine(pred, mat, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderValue=0.)
    pred = np.array(pred > 0.35)
    # Matting Image
    # mat_img = ori_img * pred[:,:,None]
    # cv2.imwrite(filename=save_path.replace('demo', 'demo_matting'), img=mat_img)
    # Mask
    pred = np.array(pred*255, dtype=np.uint8)
    cv2.imwrite(filename=save_path, img=pred)




def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='config/general_inference.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--img',
                        type=str,
                        help='imgs path')
    parser.add_argument('--exp',
                        type=str,
                        help='expression of target objects')
    parser.add_argument('--sp',
                        type=str,
                        default='demo.png',
                        help='save path')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    cfg.img = args.img
    cfg.exp = args.exp
    cfg.sp = args.sp
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.input_size = (args.input_size, args.input_size)
    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    image_path = args.img
    text_prompt = args.exp

    ori_img = cv2.imread(image_path)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[:2]
    mat, mat_inv = getTransformMat(img_size, args.input_size, True)
    img = cv2.warpAffine(
            img,
            mat,
            args.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
    img = convert(img)
    save_path = args.sp
    params = {
        'inverse': mat_inv,
        'ori_size': np.array(img_size)
    }
    _ = run_demo(args, model, img, text_prompt, save_path, ori_img, params)


if __name__ == '__main__':
    main()
