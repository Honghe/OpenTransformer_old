# -*- coding: utf-8 -*-
import glob
import os

# preprocess Aishell Dataset
root_dir = '/home/ubuntu/Data/asr/data_aishell/wav'
transcript_path = os.path.join(root_dir, '../transcript/aishell_transcript_v0.8.txt')
dirs = ['train', 'dev', 'test']

# train中有320个音频没有文本转录
# dev中有5个音频没有文本转录
# test中有0个音频没有文本转录
for d in dirs:
    data_dir = os.path.join(root_dir, d)

    transcript_dict = {}
    with open(transcript_path) as fc:
        for line in fc.readlines():
            line = line.strip()
            idx, trans = line.split(maxsplit=1)
            # 处理成以字为单位，用空格分开
            trans = ' '.join([x for x in trans if not x.isspace()])
            transcript_dict[idx] = trans

    character_path = os.path.join(data_dir, 'character')
    scp_path = os.path.join(data_dir, 'wav.scp')

    with open(character_path, 'w') as fc, open(scp_path, 'w') as fs:
        for p in glob.glob(data_dir + '/*/*.wav'):
            basename = os.path.splitext(os.path.basename(p))[0]
            if basename in transcript_dict:
                fc.write('{} {}\n'.format(basename, transcript_dict[basename]))
                fs.write('{} {}\n'.format(basename, p))
            else:
                print('warning {}'.format(basename))
