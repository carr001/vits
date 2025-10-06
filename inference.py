#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VITS Inference Script
Extracted from inference.ipynb
"""

import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def setup_environment():
    """Setup environment variables"""
    os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ['PATH']


def lj_speech_inference():
    """LJ Speech model inference"""
    print("=== LJ Speech Inference ===")
    
    # Load hyperparameters
    hps = utils.get_hparams_from_file("./configs/ljs_base.json")
    
    # Initialize model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()
    
    # Load checkpoint
    _ = utils.load_checkpoint("/Users/cxhui/PycharmProjects/AudioProcess/Projects/vits/checkpoints/pretrained_ljs.pth", net_g, None)
    
    # Inference
    try:
        stn_tst = get_text("VITS is Awesome!", hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        
        # Save audio
        output_path = "lj_speech_output.wav"
        write(output_path, hps.data.sampling_rate, audio)
        print(f"Audio saved to: {output_path}")
        
        return audio, hps.data.sampling_rate
        
    except Exception as e:
        print(f"Error in LJ Speech inference: {e}")
        return None, None


def vctk_inference():
    """VCTK multi-speaker model inference"""
    print("=== VCTK Inference ===")
    
    # Load hyperparameters
    hps = utils.get_hparams_from_file("./configs/vctk_base.json")
    
    # Initialize model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = net_g.eval()
    
    # Load checkpoint
    _ = utils.load_checkpoint("/Users/cxhui/PycharmProjects/AudioProcess/Projects/vits/checkpoints/pretrained_vctk.pth", net_g, None)
    
    # Inference
    try:
        stn_tst = get_text("VITS is Awesome!", hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            sid = torch.LongTensor([4])  # Speaker ID
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        
        # Save audio
        output_path = "vctk_output.wav"
        write(output_path, hps.data.sampling_rate, audio)
        print(f"Audio saved to: {output_path}")
        
        return audio, hps.data.sampling_rate
        
    except Exception as e:
        print(f"Error in VCTK inference: {e}")
        return None, None


def voice_conversion():
    """Voice conversion using VCTK model"""
    print("=== Voice Conversion ===")
    
    # Load hyperparameters
    hps = utils.get_hparams_from_file("./configs/vctk_base.json")
    
    # Initialize model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = net_g.eval()
    
    # Load checkpoint
    _ = utils.load_checkpoint("/Users/cxhui/PycharmProjects/AudioProcess/Projects/vits/checkpoints/pretrained_vctk.pth", net_g, None)
    
    try:
        # Load data
        dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        collate_fn = TextAudioSpeakerCollate()
        loader = DataLoader(dataset, num_workers=8, shuffle=False,
            batch_size=1, pin_memory=True,
            drop_last=True, collate_fn=collate_fn)
        data_list = list(loader)
        
        if not data_list:
            print("No data found for voice conversion")
            return
            
        # Get first data sample
        with torch.no_grad():
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x for x in data_list[0]]
            sid_tgt1 = torch.LongTensor([1])
            sid_tgt2 = torch.LongTensor([2])
            sid_tgt3 = torch.LongTensor([4])
            
            audio1 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data.cpu().float().numpy()
            audio2 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt2)[0][0,0].data.cpu().float().numpy()
            audio3 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt3)[0][0,0].data.cpu().float().numpy()
        
        # Save audio files
        write("original.wav", hps.data.sampling_rate, y[0].cpu().numpy())
        write("converted_sid_1.wav", hps.data.sampling_rate, audio1)
        write("converted_sid_2.wav", hps.data.sampling_rate, audio2)
        write("converted_sid_4.wav", hps.data.sampling_rate, audio3)
        
        print("Voice conversion completed. Audio files saved:")
        print("- original.wav")
        print("- converted_sid_1.wav")
        print("- converted_sid_2.wav")
        print("- converted_sid_4.wav")
        
    except Exception as e:
        print(f"Error in voice conversion: {e}")


def main():
    """Main function to run inference"""
    setup_environment()
    
    print("VITS Inference Script")
    print("=====================")
    
    # Run LJ Speech inference
    lj_speech_inference()
    print()
    
    # Run VCTK inference
    vctk_inference()
    print()
    
    # Run voice conversion
    voice_conversion()
    print()
    
    print("Inference completed!")


if __name__ == "__main__":
    main()