from glob import glob
from alive_progress import alive_it
from utils import read_pickle, save_pickle, draw_point, build_vocab
from torchtext.data.utils import get_tokenizer
from statistics import mean

import cv2
import os
import torch

def gather_data(data_dir):
    imgs = glob(data_dir + "/image/*")

    dct = {}

    for img in alive_it(imgs):
        img_name = os.path.basename(img).replace(".jpg", "")

        lbls = glob(data_dir + f"/grasp_label/{img_name}*")
        inss = glob(data_dir + f"/grasp_instructions/{img_name}*")

        dct[img_name] = {'label' : lbls, 'ins' : inss}
    
    return dct

def obj_cnt(dct):
    lst = []
    for _, item in alive_it(dct.items()):
        lst.append(len(item['label']))
    return lst

if __name__ == "__main__":
    root = "/".join(__file__.split("/")[:-1]) + "/src"

    train_dct_path = "/".join(__file__.split("/")[:-1]) + "/train.pickle"
    valid_dct_path = "/".join(__file__.split("/")[:-1]) + "/valid.pickle"

    train_dir = root + "/seen"
    valid_dir = root + "/unseen"

    if not os.path.exists(train_dct_path):
        train_dct = gather_data(train_dir)
        save_pickle(train_dct, train_dct_path)
    else:
        train_dct = read_pickle(train_dct_path)
    
    if not os.path.exists(valid_dct_path):
        valid_dct = gather_data(valid_dir)
        save_pickle(valid_dct, valid_dct_path)
    else:
        valid_dct = read_pickle(valid_dct_path)
    
    print(f"TOTAL TRAIN: {len(train_dct)}")
    print(f"TOTAL VALID: {len(valid_dct)}")

    train_obj = obj_cnt(train_dct)
    valid_obj = obj_cnt(valid_dct)

    print(f"TRAIN OBJECT - MAX: {max(train_obj)} - MIN: {min(train_obj)}")
    print(f"VALID OBJECT - MAX: {max(valid_obj)} - MIN: {min(valid_obj)}")

    sample_sha = '0a5db147a485709d4a8eec80837d8bf9b30a87eebdfe2e40668d0b1ea84541c5'
    sample_img = train_dir + f"/image/{sample_sha}.jpg"
    sample_lbls = sorted(train_dct[sample_sha]['label'])
    sample_inss = sorted(train_dct[sample_sha]['ins'])

    img = cv2.imread(sample_img)

    print(f"IMG SHAPE: {img.shape}")

    for idx, sample_lbl in enumerate(sample_lbls):
        data = torch.load(sample_lbl)
        quality = data[:, 0]
        ext_data = data[quality.argmax()].tolist()
        print(f"IDX: {idx} - QLY: {ext_data[0]} - CENTER: ({ext_data[1]}, {ext_data[2]}) - W/H: {ext_data[3]}, {ext_data[4]} - AN: {ext_data[-1]}")
        img = draw_point(img, ext_data)
    
    for idx, sample_ins in enumerate(sample_inss):
        ins = read_pickle(sample_ins)
        print(f"IDX: {idx} - INS: {ins}")
    
    sdr_img_path = "/".join(__file__.split("/")[:-1]) + "/sample.jpg"
    cv2.imwrite(sdr_img_path, img)

    tokenizer = get_tokenizer("basic_english")
    utidct = {**train_dct, **valid_dct}
    print(f"TOTAL SAMPLES: {len(utidct)}")
    inss = []
    toks = []
    tokslen = []
    for key, item in alive_it(utidct.items()):
        inss += item['ins']
    for ins in alive_it(inss):
        txt = read_pickle(ins)
        tokens = tokenizer(txt)
        tokslen.append(len(tokens))
        toks += tokens
    
    print(f"MAX TOKEN LEN: {max(tokslen)}")
    print(f"MIN TOKEN LEN: {min(tokslen)}")
    print(f"AVG TOKEN LEN: {mean(tokslen)}")

    vocab_path = "/".join(__file__.split("/")[:-1]) + '/vocab.pkl'
    if not os.path.exists(vocab_path):
        print("BUILDING VOCAB...")
        vocab = build_vocab(toks, tokenizer)
        save_pickle(vocab, vocab_path)
    else:
        vocab = read_pickle(vocab_path)
    
    for idx, sample_ins in enumerate(sample_inss):
        ins = read_pickle(sample_ins)
        toks = [str(vocab[x]) for x in tokenizer(ins)]
        toktxt = ' '.join(toks)
        print(f"IDX: {idx} - INS: {ins} - TOKENS: {toktxt}")
    
    angles = []
    for _, item in alive_it(train_dct.items()):
        for sublbl in item['label']:
            data = torch.load(sublbl)
            quality = data[:, 0]
            ext_data = data[quality.argmax()].tolist()
            angle = ext_data[-1]
            angles.append(angle)
    
    print(f"MAX ANGLE: {max(angles)}")
    print(f"MIN ANGLE: {min(angles)}")
    print(f"AVG ANGLE: {mean(angles)}")