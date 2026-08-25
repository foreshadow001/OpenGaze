"""训练器：训练与测试循环

输出统一写入 exp/expNN，由 utils.logger.ExperimentLogger 管理（ckpt、tensorboard、文本日志）。
所有日志经 utils.logger.get_logger 分级输出（规范见 CLAUDE.md）。
checkpoint 格式遵循 opengaze-ckpt v1 标准（见 README「Checkpoint 格式标准」）。
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm

from models import build_model
from utils.logger import get_logger
from utils.metrics import AverageMeter, angular_error

CKPT_FORMAT_VERSION = 1


class Trainer:
    def __init__(self, config, logger=None):
        """
        Args:
            config: 合并后的完整配置（dataset ∪ method）
            logger: ExperimentLogger，训练/续训必传；测试不传时 ckpt 定位退回相对路径
        """
        self.config = config
        self.logger = logger
        self.log = get_logger(__name__)

        train_cfg = config.method.train
        output_cfg = config.method.output
        self.batch_size = train_cfg.batch_size
        self.epochs = train_cfg.epochs
        # 每 N 个 epoch 保存一次 checkpoint（节约空间；最后一个 epoch 始终保存，
        # 保证训练完成必有最终 ckpt，且 resume / --test 取最新的语义不变）
        self.ckpt_save_interval = getattr(train_cfg, 'ckpt_save_interval', 5)
        self.device = torch.device(
            'cuda' if output_cfg.use_gpu and torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.train_iter = 0

        # build model
        self.model = build_model(config.method.model).to(self.device)
        self.log.info(f'[*] Number of model parameters: '
                      f'{sum(p.data.nelement() for p in self.model.parameters()):,}')
        self.log.info(f'[*] Device: {self.device}')

        # optimizer & scheduler
        if train_cfg.optimizer != 'adam':
            raise NotImplementedError(
                f'暂只支持 adam 优化器，配置为: {train_cfg.optimizer}')
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_cfg.init_lr)
        if train_cfg.lr_scheduler == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=train_cfg.lr_patience,
                                    gamma=train_cfg.lr_decay_factor)
        elif train_cfg.lr_scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        else:
            raise NotImplementedError(
                f'未知 lr_scheduler: {train_cfg.lr_scheduler}，可选 step / cosine')

        # loss
        self.loss_fn = {'l1': F.l1_loss, 'mse': F.mse_loss}[train_cfg.loss]

    # ------------------------------------------------------------------ train
    def train(self, train_loader):
        num_train = len(train_loader.dataset)
        if self.start_epoch >= self.epochs:
            self.log.info(f'[*] 已训练到 epoch {self.start_epoch}/{self.epochs}，'
                          f'无需续训（如需延长：--set method.train.epochs=N）')
            if self.logger is not None:
                self.logger.close()
            return
        self.log.info(f'[*] Train on {num_train} samples, '
                      f'{len(train_loader)} batches / epoch, '
                      f'from epoch {self.start_epoch + 1}/{self.epochs}')

        for epoch in range(self.start_epoch, self.epochs):
            self.log.info(f'Epoch: {epoch + 1}/{self.epochs}')
            self.model.train()
            self.train_one_epoch(epoch, train_loader)

            # 每 ckpt_save_interval 个 epoch 保存一次；最后一个 epoch 始终保存
            if (epoch + 1) % self.ckpt_save_interval == 0 or epoch + 1 == self.epochs:
                self.save_checkpoint({
                    'format_version': CKPT_FORMAT_VERSION,
                    'epoch': epoch + 1,
                    'train_iter': self.train_iter,
                    'model_state': self.model.state_dict(),
                    'optim_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                }, add='epoch_' + str(epoch))
            self.scheduler.step()

        if self.logger is not None:
            self.logger.close()

    def train_one_epoch(self, epoch, data_loader):
        """带 tqdm 进度条的单 epoch 训练"""
        errors = AverageMeter()
        losses_gaze = AverageMeter()
        t_epoch = time.time()
        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f'Epoch {epoch + 1}/{self.epochs}',
                    ncols=120, unit='batch')
        for _, (input_img, target) in pbar:
            input_var = input_img.float().to(self.device)
            target_var = target.float().to(self.device)

            pred_gaze = self.model(input_var)

            gaze_error_batch = np.mean(angular_error(
                pred_gaze.detach().cpu().numpy(), target_var.detach().cpu().numpy()))
            errors.update(gaze_error_batch.item(), input_var.size(0))

            loss_gaze = self.loss_fn(pred_gaze, target_var)
            self.optimizer.zero_grad()
            loss_gaze.backward()
            self.optimizer.step()
            losses_gaze.update(loss_gaze.item(), input_var.size(0))

            pbar.set_postfix({
                'loss': f'{losses_gaze.avg:.5f}',
                'error': f'{errors.avg:.3f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
            })

            self.train_iter += 1

        # epoch 结束记录一次（每 epoch 恰一个点，不依赖 print_freq 与迭代数的关系）
        self.log.info(f'epoch {epoch + 1}/{self.epochs} done in '
                      f'{(time.time() - t_epoch) / 60:.1f} min | '
                      f'train error: {errors.avg:.3f} | '
                      f'loss_gaze: {losses_gaze.avg:.5f}')
        if self.logger is not None:
            self.logger.add_scalar('Loss/gaze', losses_gaze.avg, epoch + 1)
            self.logger.add_scalar('Error/train', errors.avg, epoch + 1)
        return errors.avg, losses_gaze.avg

    # ------------------------------------------------------------------- test
    def test(self, test_loader, ckpt_path):
        """加载 checkpoint 并在测试集上评测，结果写入 exp 目录 test_result.json"""
        self.load_checkpoint(ckpt_path)

        num_test = len(test_loader.dataset)
        self.log.info(f'[*] Testing on {num_test} samples')

        errors = AverageMeter()
        losses_gaze = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),
                        desc='Testing', ncols=120, unit='batch')
            for _, (input_img, target) in pbar:
                input_var = input_img.float().to(self.device)
                target_var = target.float().to(self.device)
                pred_gaze = self.model(input_var)

                gaze_error_batch = np.mean(angular_error(
                    pred_gaze.cpu().numpy(), target_var.cpu().numpy()))
                errors.update(gaze_error_batch.item(), input_var.size(0))
                loss_gaze = self.loss_fn(pred_gaze, target_var)
                losses_gaze.update(loss_gaze.item(), input_var.size(0))
                pbar.set_postfix({'error': f'{errors.avg:.3f}'})

        self.log.info('-' * 68)
        self.log.info(f'test error: {errors.avg:.3f} - loss_gaze: '
                      f'{losses_gaze.avg:.5f} ({num_test} samples)')

        # 结果留档到实验目录（按测试数据集命名，跨数据集评测互不覆盖）
        result_name = f'test_result_{self.config.dataset.name}.json'
        if self.logger is not None:
            result_path = os.path.join(self.logger.exp_dir, result_name)
        else:
            result_path = os.path.normpath(os.path.join(
                os.path.dirname(ckpt_path), '..', result_name))
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'ckpt': os.path.abspath(ckpt_path),
                'test_dataset': self.config.dataset.name,
                'num_samples': num_test,
                'gaze_error_deg': round(errors.avg, 4),
                'loss_gaze': round(losses_gaze.avg, 6),
            }, f, indent=2, ensure_ascii=False)
        self.log.info(f'test result saved to: {result_path}')
        return errors.avg

    # -------------------------------------------------------------- checkpoint
    @property
    def _ckpt_dir(self):
        if self.logger is not None:
            return self.logger.ckpt_dir
        return '.'

    def save_checkpoint(self, state, add=None):
        """按 opengaze-ckpt v1 标准保存（见 README「Checkpoint 格式标准」）"""
        filename = (add + '_ckpt.pth') if add is not None else 'ckpt.pth'
        ckpt_path = os.path.join(self._ckpt_dir, filename)
        torch.save(state, ckpt_path)
        self.log.info(f'save checkpoint: {ckpt_path}')

    def load_checkpoint(self, input_file_path, resume=False, is_strict=True):
        """加载 checkpoint（仅支持 opengaze-ckpt v1 格式）。

        Args:
            resume: True 时完整恢复训练状态（model/optim/scheduler/start_epoch/
                    train_iter）；False 时仅加载模型权重（测试/推理用）
        """
        self.log.info(f'load checkpoint: {input_file_path}')
        ckpt = torch.load(input_file_path, map_location=self.device,
                          weights_only=True)

        # 第一道关卡：格式版本（其他项目的 ckpt 一律拒绝，不猜测兼容）
        version = ckpt.get('format_version')
        if version != CKPT_FORMAT_VERSION:
            raise KeyError(f'非 opengaze-ckpt v{CKPT_FORMAT_VERSION} 格式 '
                           f'(format_version={version})，本项目不兼容其他格式')

        try:
            self.model.load_state_dict(ckpt['model_state'], strict=is_strict)
        except KeyError as e:
            raise KeyError(
                f'非 opengaze-ckpt v{CKPT_FORMAT_VERSION} 格式（缺字段 {e}）') from e

        if resume:
            try:
                self.optimizer.load_state_dict(ckpt['optim_state'])
                self.scheduler.load_state_dict(ckpt['scheduler_state'])
                self.start_epoch = ckpt['epoch']
                self.train_iter = ckpt['train_iter']
            except KeyError as e:
                raise KeyError(
                    f'断点续训要求完整 v1 字段（缺字段 {e}）') from e
            self.log.info(
                f'[*] Resumed from {input_file_path}: '
                f'next epoch {self.start_epoch + 1}/{self.epochs}, '
                f'train_iter {self.train_iter}, '
                f'lr {self.optimizer.param_groups[0]["lr"]:.6f}')
        else:
            self.log.info(f'[*] Loaded {input_file_path} checkpoint (weights only)')
