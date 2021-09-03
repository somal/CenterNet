import argparse
import os

import torch
from torch.utils import data

from src.lib.datasets.dataset.coco_cl import MultipleAnnotationsCOCOCL
from src.lib.datasets.dataset_factory import get_dataset
from src.lib.logger import Logger
from src.lib.models.model import create_model, load_model, save_model
from src.lib.opts import opts
from src.lib.trains.train_factory import train_factory

coco_annotation_folders = ('esaul_20', 'esaul_21', 'raspd-2_30', 'rcocs-1_12')


def build_dataset(dataset_cls, opt: argparse.Namespace, split):
    if opt.dataset == 'coco_cl' and opt.task == 'ctdet':
        new_dataset_instance = MultipleAnnotationsCOCOCL.build(dataset_cls, opt, split, coco_annotation_folders)
        return new_dataset_instance
    else:
        return dataset_cls(opt, split)


def main(opt: argparse.Namespace):
    opt.data_dir = '/home/msokolov/PycharmProjects/bd_presales/cable_line/datasets/'

    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(Dataset, opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if opt.test:
        print('Start testing')
        _, preds = trainer.val(0, val_loader)
        # val_loader.dataset.run_eval(preds, opt.save_dir)
        [d.run_eval(preds, opt.save_dir) for d in val_loader.dataset.datasets]
        return

    train_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(Dataset, opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()  # type: argparse.Namespace
    main(opt)
