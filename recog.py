import argparse
import os

import torch
import torchaudio
import yaml

from otrans.data import load_vocab, normalization, compute_fbank
from otrans.model import Transformer
from otrans.recognizer import TransformerRecognizer


def calc_fbank(wav_file, params):
    wavform, sample_frequency = torchaudio.load_wav(wav_file)
    feature = compute_fbank(wavform, num_mel_bins=params['num_mel_bins'], sample_frequency=sample_frequency)

    # if params['apply_cmvn']:
    #     spk_id = self.utt2spk[utt_id]
    #     stats = kio.load_mat(self.cmvns[spk_id])
    #     feature = apply_cmvn(feature, stats)

    if params['normalization']:
        feature = normalization(feature)

    feature = torch.stack((feature,))
    feature_length = torch.tensor([1])
    return feature, feature_length


def main(args):
    checkpoint = torch.load(args.load_model)
    if 'params' in checkpoint:
        params = checkpoint['params']
    else:
        assert os.path.isfile(args.config), 'please specify a configure file.'
        with open(args.config, 'r') as f:
            params = yaml.load(f)

    params['data']['shuffle'] = False
    params['data']['spec_argument'] = False
    params['data']['short_first'] = False
    params['data']['batch_size'] = args.batch_size

    model = Transformer(params['model'])

    model.load_state_dict(checkpoint['model'])
    print('Load pre-trained model from %s' % args.load_model)

    model.eval()
    if args.ngpu > 0:
        model.cuda()

    char2unit = load_vocab(params['data']['vocab'])
    unit2char = {i: c for c, i in char2unit.items()}

    recognizer = TransformerRecognizer(model, unit2char=unit2char, beam_width=args.beam_width,
                                       max_len=args.max_len, penalty=args.penalty, lamda=args.lamda, ngpu=args.ngpu)

    # inputs_length: [len]
    inputs, inputs_length = calc_fbank(args.file, params['data'])
    if args.ngpu > 0:
        inputs = inputs.cuda()
        inputs_length = inputs_length.cuda()

    preds = recognizer.recognize(inputs, inputs_length)
    print('preds: {}'.format(preds[0].replace(' ', '')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-bw', '--beam_width', type=int, default=5)
    parser.add_argument('-p', '--penalty', type=float, default=0.6)
    parser.add_argument('-ld', '--lamda', type=float, default=5)
    parser.add_argument('-m', '--load_model', type=str, default=None)
    parser.add_argument('-ml', '--max_len', type=int, default=100)
    parser.add_argument('-s', '--suffix', type=str, default=None)
    parser.add_argument('-f', '--file', type=str, default=None)
    cmd_args = parser.parse_args()

    main(cmd_args)
