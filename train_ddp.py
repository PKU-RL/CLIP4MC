from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import os
import json
import time
import argparse
import datetime
from tensorboardX import SummaryWriter
from torch.amp import autocast

from model import NaiveCLIP, CLIP4MC, CLIP4MC_simple

from module import get_optimizer
from data import get_naive_dataloader
from utils import get_logger, set_seed, compute_metrics

torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(0, 18000))
_time = time.strftime("%y_%m_%d_%H:%M:%S", time.localtime())
local_rank = torch.distributed.get_rank()
n_gpu = torch.distributed.get_world_size()

assert local_rank == int(os.environ['LOCAL_RANK']), \
    "local_rank {} is not equal to os.environ['LOCAL_RANK'] {}".format(local_rank, os.environ['LOCAL_RANK'])
assert n_gpu == int(os.environ['WORLD_SIZE']), \
    "n_gpu {} is not equal to os.environ['WORLD_SIZE'] {}".format(n_gpu, os.environ['WORLD_SIZE'])

output_dir = os.path.join('./ckpt', _time)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

log_file = os.path.join(output_dir, 'log.txt')
logger = get_logger(log_file)


def get_args(description='MineCLIP args'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--n_display', type=int, default=10, help='display step')

    parser.add_argument('--dataset_log_file', type=str, default='./Youtube_dataset/log.json',
                        help='dataset log file of data paths')
    parser.add_argument('--use_pretrained_CLIP', action='store_true', default=False, help='use pretrained CLIP model')
    parser.add_argument('--pretrain_CLIP_path', type=str, default="./ViT-B-16.pt", help='pretrained CLIP model path')
    parser.add_argument('--use_pretrained_model', action='store_true', default=False, help='use pretrained model')
    parser.add_argument('--pretrain_model_path', type=str, default="./CLIP4MC.pt", help='pretrained model path')
    parser.add_argument('--model_type', type=str, default='CLIP4MC', choices=['CLIP4MC', 'CLIP4MC_simple', 'MineCLIP'])

    parser.add_argument('--clip_frame_num', type=int, default=16, help='frame num for each shorter clip')
    parser.add_argument('--clip_frame_stride', type=int, default=8, help='frame stride for each shorter clip')

    parser.add_argument('--use_mask', action='store_true', default=False, help='data process name')
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='num workers')
    parser.add_argument('--batch_size_eval', type=int, default=400, help='batch size evaluate')

    parser.add_argument('--epochs', type=int, default=40, help='epochs')
    parser.add_argument('--optimizer_name', type=str, default="BertAdam", help='optimizer name')
    parser.add_argument('--schedule_name', type=str, default="warmup_cosine", help='schedule name')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='initial learning rate')
    parser.add_argument('--layer_wise_lr_decay', type=float, default=0.65, help='coefficient for bert branch.')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='Learning rate exp epoch decay')
    parser.add_argument("--warmup_proportion", default=0.005, type=float, help="Warmup proportion")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')

    parser.add_argument('--text_freeze_layer', type=int, default=11, help='text encoder freeze layer')
    parser.add_argument('--video_freeze_layer', type=int, default=11, help='video encoder freeze layer')

    args = parser.parse_args()
    args.seed = args.seed + local_rank
    return args


def save_model(epoch, model, type_name=""):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)


def train_epoch(epoch, args, model, train_dataloader, device, optimizer, scheduler, global_step):
    global logger
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    grad_step = 0

    for step, batch in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        batch = tuple(t.to(device) for t in batch)
        with autocast(device_type='cuda'):
            loss = model(*batch, train=True)

        loss.backward()

        total_loss += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scheduler is not None:
            scheduler.step()

        optimizer.step()
        optimizer.zero_grad()

        model.module.clamp_logit_scale()

        global_step += 1
        grad_step += 1
        if global_step % log_step == 0 and local_rank == 0:
            logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                        args.epochs, step + 1,
                        len(train_dataloader),
                        "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                        float(loss),
                        (time.time() - start_time) / log_step)
            start_time = time.time()

    total_loss = total_loss / grad_step
    return total_loss, global_step


def eval_epoch(model, test_dataloader, writer, epoch, device):
    model.eval()

    batch_list_t = []
    batch_list_v = []
    with torch.no_grad():

        for bid, batch in enumerate(test_dataloader):
            torch.cuda.empty_cache()
            batch = tuple(t.to(device) for t in batch)
            with autocast(device_type='cuda'):
                video_features, text_features = model(*batch, train=False)
            if local_rank == 0:
                if isinstance(video_features, list):
                    if len(batch_list_v) == 0:
                        batch_list_v = [[] for _ in range(len(video_features))]
                    for i in range(len(video_features)):
                        batch_list_v[i].append(video_features[i].cpu())
                else:
                    batch_list_v.append(video_features.cpu())
                if isinstance(text_features, list):
                    if len(batch_list_t) == 0:
                        batch_list_t = [[] for _ in range(len(text_features))]
                    for i in range(len(text_features)):
                        batch_list_t[i].append(text_features[i].cpu())
                else:
                    batch_list_t.append(text_features.cpu())

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        if local_rank == 0:
            if isinstance(batch_list_v[0], list):
                kind = len(batch_list_v)
                video_features = [torch.cat(itm, dim=0) for itm in batch_list_v]
            else:
                kind = 1
                video_features = torch.cat(batch_list_v, dim=0)
            if isinstance(batch_list_t[0], list):
                if kind == 1:
                    kind = len(batch_list_t)
                else:
                    assert kind == len(batch_list_t)
                text_features = [torch.cat(itm, dim=0) for itm in batch_list_t]
            else:
                text_features = torch.cat(batch_list_t, dim=0)

            final_sim_matrix = 0

            for ki in range(kind):
                sub_video_features = video_features[ki] if isinstance(video_features, list) else video_features
                sub_text_features = text_features[ki] if isinstance(text_features, list) else text_features
                sim_matrix = sub_video_features @ sub_text_features.t()
                final_sim_matrix += sim_matrix

            vt_metrics = compute_metrics(final_sim_matrix.cpu().numpy())
            tv_metrics = compute_metrics(final_sim_matrix.cpu().numpy().T)

            logger.info("Video-to-Text:")
            logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f}'
                        ' - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                        format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'],
                               vt_metrics['MedianR'], vt_metrics['MeanR']))
            logger.info("Text-to-Video:")
            logger.info('\t>>>  T2V$R@1: {:.1f} - T2V$R@5: {:.1f} - T2V$R@10: {:.1f}'
                        ' - T2V$Median R: {:.1f} - T2V$Mean R: {:.1f}'.
                        format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'],
                               tv_metrics['MedianR'], tv_metrics['MeanR']))

            for k, v in tv_metrics.items():
                writer.add_scalar("V2T_{}/{}".format('all', k), v, epoch)
            for k, v in vt_metrics.items():
                writer.add_scalar("T2V_{}/{}".format('all', k), v, epoch)
            writer.flush()


def main(args):
    global logger

    # Setup CUDA, GPU & distributed training
    set_seed(args.seed)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Setup logging
    if local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("\t{}: {}".format(key, args.__dict__[key]))

        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        os.system('cp {} {}'.format('config.yaml', output_dir))

        writer_dir = os.path.join(output_dir, 'runs')
        logger.info("writer_dir: {}".format(writer_dir))
        train_writer = SummaryWriter(os.path.join(writer_dir, 'train'))
        test_writer = SummaryWriter(os.path.join(writer_dir, 'test'))
    else:
        train_writer = None
        test_writer = None

    # Setup Model
    if args.use_pretrained_CLIP and not args.use_pretrained_model:
        if local_rank == 0:
            logger.info("Loading pretrained clip from {}".format(args.pretrain_CLIP_path))
        pretrained_clip = torch.jit.load(args.pretrain_CLIP_path)
    else:
        pretrained_clip = None

    if args.model_type == 'CLIP4MC':
        model = CLIP4MC(frame_num=args.clip_frame_num,
                        share_sequence_encoder=True,
                        share_adapter=True,
                        use_action=False,
                        use_brief_text=False,
                        pretrained_clip=pretrained_clip, )
    elif args.model_type == 'CLIP4MC_simple':
        model = CLIP4MC_simple(frame_num=args.clip_frame_num,
                               share_sequence_encoder=True,
                               share_adapter=True,
                               use_action=False,
                               use_brief_text=False,
                               pretrained_clip=pretrained_clip)
    elif args.model_type == 'MineCLIP':
        model = NaiveCLIP(frame_num=args.clip_frame_num,
                          pretrained_clip=pretrained_clip)
    else:
        raise NotImplementedError

    if args.use_pretrained_model:
        if local_rank == 0:
            logger.info("Loading pretrained model from {}".format(args.pretrain_model_path))
        model.load_state_dict(torch.load(args.pretrain_model_path))

    model = model.to(device)

    # Setup dataset
    train_dataloader, train_sampler, train_length \
        = get_naive_dataloader(args.dataset_log_file, args.use_mask, args.batch_size, 'train', args.num_workers)
    test_dataloader, test_sampler, test_length \
        = get_naive_dataloader(args.dataset_log_file, args.use_mask, args.batch_size_eval, 'test', args.num_workers)

    num_train_optimization_steps = train_length // args.batch_size * args.epochs

    if local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_eval)
        logger.info("  Num steps = %d", len(test_dataloader))

    # Prepare optimizer
    optimizer = get_optimizer(optimizer_name=args.optimizer_name,
                              schedule_name=args.schedule_name,
                              model=model,
                              lr=args.lr,
                              layer_wise_lr_decay=args.layer_wise_lr_decay,
                              weight_decay=args.weight_decay,
                              warmup_proportion=args.warmup_proportion,
                              t_total=num_train_optimization_steps,
                              max_grad_norm=args.max_grad_norm,
                              text_freeze_layer=args.text_freeze_layer,
                              video_freeze_layer=args.video_freeze_layer)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    scheduler = None

    # Train!
    global_step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, optimizer,
                                           scheduler, global_step)
        if local_rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            train_writer.add_scalar('loss', tr_loss, epoch + 1)
            save_model(epoch, model, type_name="")

        if local_rank == 0:
            logger.info("Eval on test dataset")
        eval_epoch(model, test_dataloader, test_writer, epoch, device)


if __name__ == "__main__":
    main(get_args())
