import tensorflow as tf
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=True,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/kss_text_train_filelist.txt',
        validation_files='filelists/kss_text_val_filelist.txt',
        text_cleaners=['korean_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=64,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams


def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_directory', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('-d', '--dataset-path', type=str,
                        default='./', help='Path to dataset')
    # parser.add_argument('-d', '--dataset-path', type=str,
    #                     default='./', help='Path to dataset')
    # parser.add_argument('--log-file', type=str, default='nvlog.json',
    #                     help='Filename for logging')
    # parser.add_argument('--anneal-steps', nargs='*',
    #                     help='Epochs after which decrease learning rate')
    # parser.add_argument('--anneal-factor', type=float, choices=[0.1, 0.3], default=0.1,
    #                     help='Factor for annealing learning rate')

    #                      type=str, help='Path to configuration file')
    parser.add_argument('--seed', default=1234, type=int,
                        help='Seed for random number generators')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=500,
                          help='Number of total epochs to run')
    training.add_argument('--iters-per-checkpoint', type=int, default=500,
                          help='Number of epochs per checkpoint')
    training.add_argument('--checkpoint-path', type=str, default='',
                          help='Checkpoint path to resume training')
    training.add_argument('--resume-from-last', action='store_true',
                          help='Resumes training from the last checkpoint; uses the directory provided with \'--output\' option to search for the checkpoint \"checkpoint_<model_name>_last.pt\"')
    training.add_argument('--dynamic-loss-scaling', type=bool, default=True,
                          help='Enable dynamic loss scaling')
    training.add_argument('--amp', action='store_true',
                          help='Enable AMP')
    training.add_argument('--fp16-run', action='store_true', default=False,
                          help='Enable AMP')
    training.add_argument('--cudnn-enabled', action='store_true', default=True,
                          help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', action='store_true', default=False,
                          help='Run cudnn benchmark')
    training.add_argument('--ignore-layers', action='store_true', default=['embedding.weight'],
                          )
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')
    training.add_argument('--n-symbols', action='store_true', default=len(symbols),
                     help='symbols length')
    training.add_argument('--symbols-embedding-dim', action='store_true', default=512,
                     help='symbols embedding dim')

    speaker = parser.add_argument_group('speaker')
    speaker.add_argument('--n-speakers', action='store_true', default=128)
    speaker.add_argument('--speakers-embedding-dim', action='store_true', default=16)

    emotion = parser.add_argument_group('emotion')
    emotion.add_argument('--use-emotions', action='store_true', default=True)
    emotion.add_argument('--n-emotions', action='store_true', default=15)
    emotion.add_argument('--emotions-embedding-dim', action='store_true', default=8)

    encoder = parser.add_argument_group('encoder')
    encoder.add_argument('--encoder-kernel-size', action='store_true', default=5)
    encoder.add_argument('--encoder-n-convolutions', action='store_true', default=3)
    encoder.add_argument('--encoder-embedding-dim', action='store_true', default=512)

    decoder = parser.add_argument_group('decoder')
    decoder.add_argument('--n-frames-per-step', action='store_true', default=1)
    decoder.add_argument('--decoder-rnn-dim', action='store_true', default=1024)
    decoder.add_argument('--prenet-dim', action='store_true', default=256)
    decoder.add_argument('--max-decoder-steps', action='store_true', default=1000)
    decoder.add_argument('--gate-threshold', action='store_true', default=0.5)
    decoder.add_argument('--p-attention-dropout', action='store_true', default=0.1)
    decoder.add_argument('--p-decoder-dropout', action='store_true', default=0.1)
    decoder.add_argument('--decoder-no-early-stopping', action='store_true', default=False)

    attention = parser.add_argument_group('attention')
    attention.add_argument('--attention-rnn-dim', action='store_true', default=1024)
    attention.add_argument('--attention-dim', action='store_true', default=128)
    attention.add_argument('--attention-location-n-filters', action='store_true', default=32)
    attention.add_argument('--attention-location-kernel-size', action='store_true', default=31)

    postnet = parser.add_argument_group('postnet')
    postnet.add_argument('--postnet-embedding-dim', action='store_true', default=512)
    postnet.add_argument('--postnet-kernel-size', action='store_true', default=5)
    postnet.add_argument('--postnet-n-convolutions', action='store_true', default=5)

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument(
        '--use-saved-learning-rate', default=False, type=bool)
    optimization.add_argument('-lr', '--learning-rate', type=float, default=1e-3,   # 5e-4
                              help='Learing rate')
    optimization.add_argument('--weight-decay', default=1e-6, type=float,
                              help='Weight decay')
    optimization.add_argument('--grad-clip-thresh', default=1.0, type=float,
                              help='Clip threshold for gradients')
    optimization.add_argument('-bs', '--batch-size', type=int, default=32,
                              help='Batch size per GPU')    # TODO default 64
    optimization.add_argument('--grad-clip', default=5.0, type=float,
                              help='Enables gradient clipping and sets maximum gradient norm value')
    optimization.add_argument('--mask-padding', default=True, type=bool)

    # dataset parameters
    dataset = parser.add_argument_group('dataset parameters')
    dataset.add_argument('--load-mel-from-disk', action='store_true', default=False,
                         help='Loads mel spectrograms from disk instead of computing them on the fly')
    dataset.add_argument('--training-files',
                         default='filelists/kss_text_train_filelist.txt',
                         type=str, help='Path to training filelist')
    dataset.add_argument('--validation-files',
                         default='filelists/kss_text_val_filelist.txt',
                         type=str, help='Path to validation filelist')
    dataset.add_argument('--text-cleaners', nargs='*',
                         default=['korean_cleaners'], type=str,
                         help='Type of text cleaners for input text')

    # audio parameters
    audio = parser.add_argument_group('audio parameters')
    audio.add_argument('--max-wav-value', default=32768.0, type=float,
                       help='Maximum audiowave value')
    audio.add_argument('--sampling-rate', default=22050, type=int,
                       help='Sampling rate')
    audio.add_argument('--filter-length', default=1024, type=int,
                       help='Filter length')
    audio.add_argument('--hop-length', default=256, type=int,
                       help='Hop (stride) length')
    audio.add_argument('--win-length', default=1024, type=int,
                       help='Window length')
    audio.add_argument('--n-mel-channels', default=80, type=int,
                       help='mel channels')
    audio.add_argument('--mel-fmin', default=0.0, type=float,
                       help='Minimum mel frequency')
    audio.add_argument('--mel-fmax', default=8000.0, type=float,
                       help='Maximum mel frequency')

    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--distributed-run', default=False, type=bool,
                             help='enable distributed run')
    distributed.add_argument('--rank', default=0, type=int,
                             help='Rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='Number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', type=str, default='tcp://localhost:54321',
                             help='Url used to set up distributed training')
    distributed.add_argument('--group-name', type=str, default='group_name',
                             required=False, help='Distributed group name')
    distributed.add_argument('--dist-backend', default='nccl', type=str, choices={'nccl'},
                             help='Distributed run backend')

    benchmark = parser.add_argument_group('benchmark')
    benchmark.add_argument('--bench-class', type=str, default='')

    return parser