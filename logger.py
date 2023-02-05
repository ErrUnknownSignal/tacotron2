import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy
from text import symbols
import numpy as np
import matplotlib.pylab as plt


# font_family = 'NanumGothic'
font_family = 'Malgun Gothic'
plt.rc('font', family=font_family)
plt.rc('axes', unicode_minus=False)


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')

    def log_fig(self, model, valset, iteration, args):
        sequence = valset.get_text("마음대로 생각해.")
        sequence = torch.autograd.Variable(torch.from_numpy(np.array(sequence)[None, :]))
        sequence = sequence.cuda().long()
        mel_outputs, mel_outputs_postnet, gate_output, alignments = model.inference(sequence)
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        axes[0].imshow(mel_outputs.float().data.cpu().numpy()[0], aspect='auto', origin='lower', interpolation='none')
        axes[1].imshow(mel_outputs_postnet.float().data.cpu().numpy()[0], aspect='auto', origin='lower', interpolation='none')
        axes[2].imshow(alignments.float().data.cpu().numpy()[0].T, aspect='auto', origin='lower', interpolation='none')
        temp = torch.sigmoid(gate_output).data.cpu().numpy()
        temp_len = len(temp[0])
        axes[3].scatter(range(temp_len), temp.reshape(temp_len))


        # https://github.com/NVIDIA/tacotron2/issues/409
        dur_frames = torch.histc(torch.argmax(alignments[0], dim=1).float(), min=0, max=sequence.shape[1]-1, bins=sequence.shape[1])    # number of frames each letter taken the maximum focus of the model.
        dur_seconds = dur_frames * (args.hop_length / args.sampling_rate)   # convert from frames to seconds
        end_times = dur_seconds * 0.0   # new empty list
        for i, dur_second in enumerate(dur_seconds):    # calculate the end times for each letter.
            end_times[i] = end_times[i-1] + dur_second  # by adding up the durations of itself and all the letters that go before it
        start_times = torch.nn.functional.pad(end_times, (1, 0))[:-1]    # calculate the start times by assuming the next letter starts the moment the last one ends.

        dur_frames = dur_frames.float().data.cpu().numpy()
        start_times = start_times.float().data.cpu().numpy()
        end_times = end_times.float().data.cpu().numpy()
        xs, hs, ws = [], [], []
        for i, start in enumerate(start_times):
            xs.append((end_times[i] + start) / 2 + 0.4)
            # hs.append(seg.score)
            hs.append(dur_frames[i])
            ws.append(end_times[i] - start)
            axes[4].annotate(symbols[sequence[0][i].item()], (start + 0.6, -0.07))
        axes[4].bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")


        fig.savefig('./outdir/' + str(iteration) + '.png')
        # plt.close(fig)

        self.add_figure(tag='epoch_{}/graph'.format(iteration), figure=fig)
