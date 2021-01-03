from __future__ import division, print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import uuid
import os
from tensorboardX import SummaryWriter
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

from dataset import Dataset
from model import Model
from config import Config, backward_compatible_config, dump_config_details_to_tensorboard
from eval_action_detection import EvalActionDetection

def save_checkpoint(best_state, ckpt_filename):
    torch.save(best_state, ckpt_filename)
    logging.info(f'Saved checkpoint in {ckpt_filename}')

def generate_experiment_dir(log_dir):
    num_trails = 10
    for _ in range(num_trails):
        timestamp = time.strftime('%Y-%m-%d-%H:%M:%S')
        experiment_dir = f'{log_dir}/{timestamp}__{uuid.uuid4()}'

        try:
            os.mkdir(experiment_dir)
            return experiment_dir
        except Exception as e:
            logging.warn(f'error making the experiment directory. {e}')
    raise Exception(f'error making the experiment directory after trying {num_trails} times.')

def dump_current_state_to_tensorboard(summary_writer, config, learning_rate, train_metrics, valid_metrics, epoch):
    summary_writer.add_scalar('learning_rate', learning_rate, epoch)
    for k,v in train_metrics.items():
        if 'gt_label_accuracy' in k:
            summary_writer.add_scalar(f'refinement_train/{k}', v, epoch)
        else:
            summary_writer.add_scalar(f'train/{k}', v, epoch)
    for k,v in valid_metrics.items():
        if 'gt_label_accuracy' in k:
            summary_writer.add_scalar(f'refinement_valid/{k}', v, epoch)
        else:
            summary_writer.add_scalar(f'valid/{k}', v, epoch)

def dump_refinement_state_to_tensorboard(summary_writer, refinement_stats, train_pseudo_gt_stats, valid_pseudo_gt_stats, epoch):
    for k,v in train_pseudo_gt_stats.items():
        summary_writer.add_scalar(f'refinement_train/{k}', v, epoch)
    for k,v in valid_pseudo_gt_stats.items():
        summary_writer.add_scalar(f'refinement_valid/{k}', v, epoch)

    iteration_number = len(refinement_stats['avg_mAP_over_iterations']) - 1
    summary_writer.add_scalar('refinement_valid/refined_avg_mAP', refinement_stats['this_iteration_avg_mAP'], iteration_number)

    if iteration_number > 0:
        summary_writer.add_scalar('refinement_valid/refined_avg_mAP_delta', refinement_stats['avg_mAP_over_iterations'][-1] - refinement_stats['avg_mAP_over_iterations'][-2], iteration_number)

def dump_best_model_metrics_to_tensorboard(summary_writer, best_state, refinement_stats):
    for k,v in best_state['train_metrics'].items():
        summary_writer.add_scalar(f'best_state/train/{k}', v, 0)
    for k,v in best_state['valid_metrics'].items():
        summary_writer.add_scalar(f'best_state/valid/{k}', v, 0)
    summary_writer.add_scalar('best_state/convergence_epoch', best_state['convergence_epoch'], 0)

    best_iteration_number = np.argmax(refinement_stats['avg_mAP_over_iterations'])
    summary_writer.add_scalar('best_state/base_avg_mAP', refinement_stats['avg_mAP_over_iterations'][0], 0)
    summary_writer.add_scalar('best_state/refined_avg_mAP', refinement_stats['avg_mAP_over_iterations'][best_iteration_number], 0)
    summary_writer.add_scalar('best_state/refined_avg_mAP_diff_from_base', refinement_stats['avg_mAP_over_iterations'][best_iteration_number] - refinement_stats['avg_mAP_over_iterations'][0], 0)
    summary_writer.add_scalar('best_state/refined_convergence_iteration', best_iteration_number, 0)

def create_datasets(config, args):
    datasets, data_loaders = {}, {}

    datasets['train'] = Dataset(features_filename=args.train_features_filename, labels_filename=args.train_labels_filename,
                                snippet_size=config['snippet_size'], stream=config['stream'])
    datasets['valid'] = Dataset(features_filename=args.valid_features_filename, labels_filename=args.valid_labels_filename,
                                snippet_size=config['snippet_size'], stream=config['stream'])
    data_loaders['train'] = DataLoader(dataset=datasets['train'], batch_size=config['batch_size'], pin_memory=True,
                                       num_workers=config['num_workers'], shuffle=True, collate_fn=datasets['train'].collate_fn)
    data_loaders['valid'] = DataLoader(dataset=datasets['valid'], batch_size=config['batch_size'], pin_memory=True,
                                       num_workers=config['num_workers'], shuffle=True, collate_fn=datasets['valid'].collate_fn)
    config['num_classes'] = datasets['train'].num_classes
    config['input_size'] = datasets['train'].feature_size

    return config, datasets, data_loaders

def process_one_batch(config, model, features, targets, device, data_loader, optimizer=None):
    cas, attention = model(features.to(device))
    data_loader.dataset.save_predictions(targets, cas, attention)
    loss_results = model.loss(cas, attention, targets, device)

    if optimizer is not None and not np.isnan(loss_results['loss'].item()):
        optimizer.zero_grad()
        loss_results['loss'].backward()
        optimizer.step()

    metrics = {k: v.item() for (k,v) in loss_results.items()}

    return metrics

def full_epoch(config, model, data_loader, device, detection_eval, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()
    model.to(device)

    accumulated_metrics = {}
    for features, targets in data_loader:
        metrics = process_one_batch(config=config, model=model, features=features, targets=targets, device=device, data_loader=data_loader, optimizer=optimizer)
        # accumulate the metrics
        for metric, value in metrics.items():
            if metric not in accumulated_metrics:
                accumulated_metrics[metric] = 0
            accumulated_metrics[metric] += value

    # compute the averaged metrics
    for metric in accumulated_metrics:
        accumulated_metrics[metric] /= len(data_loader)

    eval_results = data_loader.dataset.eval_saved_predictions()
    accumulated_metrics.update(eval_results)

    predictions_df = data_loader.dataset.get_detection_predictions(top_k_labels=config['top_k_labels'],
                                                                   min_cas_score=config['min_cas_score'],
                                                                   min_attention_score=config['min_attention_score'],
                                                                   padding=config['refinement_padding'])
    accumulated_metrics['avg_mAP'] = detection_eval.evaluate(predictions_df)*100

    return accumulated_metrics, predictions_df

def load_best_model_and_update_pseudo_gt(best_state, config, model, optimizer, scheduler, train_data_loader, valid_data_loader, train_detection_eval, valid_detection_eval, device):
    # load best model; do one forward pass to populate CAS and attenuation from the best model; update the pseudo gt
    model.load_state_dict(best_state['model'])
    full_epoch(config, model, train_data_loader, device, train_detection_eval)
    full_epoch(config, model, valid_data_loader, device, valid_detection_eval)

    train_pseudo_gt_stats = train_data_loader.dataset.update_pseudo_bg_fg_gt(pseudo_gt_generator_type=config['pseudo_gt_generator_type'],
                                                        top_k_labels=config['refinement_top_k_labels'],
                                                        min_cas_score=config['refinement_min_cas_score'],
                                                        min_attention_score=config['refinement_min_attention_score'],
                                                        padding=config['refinement_padding'],
                                                        from_m_iterations=config['refinement_from_m_iterations'],
                                                        pseudo_gt_dropout=config['pseudo_gt_dropout'])
    valid_pseudo_gt_stats = valid_data_loader.dataset.update_pseudo_bg_fg_gt(pseudo_gt_generator_type=config['pseudo_gt_generator_type'],
                                                        top_k_labels=config['refinement_top_k_labels'],
                                                        min_cas_score=config['refinement_min_cas_score'],
                                                        min_attention_score=config['refinement_min_attention_score'],
                                                        padding=config['refinement_padding'],
                                                        from_m_iterations=config['refinement_from_m_iterations'],
                                                        pseudo_gt_dropout=config['pseudo_gt_dropout'])

    logging.info(f'train_pseudo_gt_stats: {train_pseudo_gt_stats}')
    logging.info(f'valid_pseudo_gt_stats: {valid_pseudo_gt_stats}')

    # reset the model to train from scratch
    if config['reset']:
        logging.info('Resetting the model to train from scratch')
        model = Model(num_classes=config['num_classes'], input_size=config['input_size'], num_layers=config['num_layers'],
                      alpha_l1_loss=config['alpha_l1_loss'], alpha_group_sparsity_loss=config['alpha_group_sparsity_loss'],
                      alpha_pseudo_gt_loss=config['alpha_pseudo_gt_loss'],
                      weighted_psgt_loss=config['weighted_psgt_loss'],
                      pseudo_gt_loss_dim=config['pseudo_gt_loss_dim'])

        optimizer = Adam(model.parameters(), lr=config['initial_lr'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config['lr_decay'], patience=config['lr_patience'], verbose=True)
    else:
        optimizer.load_state_dict(best_state['optimizer'])
        scheduler.load_state_dict(best_state['scheduler'])

    # reset best state
    best_state['model'] = model.state_dict()
    best_state['optimizer'] = optimizer.state_dict()
    best_state['scheduler'] = scheduler.state_dict()
    best_state['train_metrics'] = {'loss': float('inf')}
    best_state['valid_metrics'] = {'loss': float('inf')}
    best_state['num_epochs_since_best_valid_loss'] = 0

    return best_state, model, optimizer, scheduler, train_data_loader, valid_data_loader, train_pseudo_gt_stats, valid_pseudo_gt_stats

def train(args, config, model, train_data_loader, valid_data_loader, device, optimizer, scheduler, initial_epoch, experiment_dir, summary_writer):
    dump_config_details_to_tensorboard(summary_writer, config)
    logging.info(f'config {config}')

    refinement_stats = {'this_iteration_avg_mAP': 0.0,
                        'avg_mAP_over_iterations': []}

    best_state = {'config': config,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'train_metrics': {'loss': float('inf')},
                  'valid_metrics': {'loss': float('inf')},
                  'convergence_epoch': 0,
                  'num_epochs_since_best_valid_loss': 0,
                  'refinement_stats': refinement_stats}

    train_detection_eval = EvalActionDetection(args.train_labels_filename, tiou_thresholds=config['tiou_thresholds'], verbose=False)
    valid_detection_eval = EvalActionDetection(args.valid_labels_filename, tiou_thresholds=config['tiou_thresholds'], verbose=False)
    model.to(device)

    epoch_offset = 0
    for iteration in range(config['refinement_iterations']):
        logging.info(f'Iteration #{iteration}')
        refinement_stats['this_iteration_avg_mAP'] = 0.0
        best_valid_predictions_df = None
        for epoch in range(initial_epoch+1, config['max_epoch']+1):
            start_time = time.time()
            train_metrics, train_predictions_df = full_epoch(config, model, train_data_loader, device, train_detection_eval, optimizer)
            valid_metrics, valid_predictions_df = full_epoch(config, model, valid_data_loader, device, valid_detection_eval)

            # reduce the learning rate according to the scheduler policy
            learning_rate = optimizer.param_groups[0]['lr']
            scheduler.step(valid_metrics['loss'])

            logging.info(f'Epoch #{epoch}: [train: {train_metrics["loss"]:.2e}, {[(k, np.round(v,2)) for (k,v) in train_metrics.items() if "loss" not in k]}]')
            logging.info(f'Epoch #{epoch}: [valid: {valid_metrics["loss"]:.2e}, {[(k, np.round(v,2)) for (k,v) in valid_metrics.items() if "loss" not in k]}]')
            logging.info(f'[lr: {learning_rate:.2e}] [time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}]')

            dump_current_state_to_tensorboard(summary_writer, config, learning_rate, train_metrics, valid_metrics, epoch_offset + epoch)

            if np.isnan(train_metrics['loss']) or np.isnan(valid_metrics['loss']):
                logging.info('got a NAN loss. Breaking!')
                break

            # save the model to disk if it has improved, or increment counter for early stopping otherwise.
            if best_state['valid_metrics']['loss'] >= valid_metrics['loss']:
                logging.info(f'Got a new best model with valid loss = {valid_metrics["loss"]:.2e} and avg_mAP = {valid_metrics["avg_mAP"]:.2f}')
                refinement_stats['this_iteration_avg_mAP'] = valid_metrics['avg_mAP']
                best_valid_predictions_df = valid_predictions_df
                best_state = {'config': config,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict(),
                              'train_metrics': train_metrics,
                              'valid_metrics': valid_metrics,
                              'convergence_epoch': epoch,
                              'num_epochs_since_best_valid_loss': 0,
                              'refinement_stats': refinement_stats}
            else:
                best_state['num_epochs_since_best_valid_loss'] += 1

            # valid loss early stopping
            if best_state['num_epochs_since_best_valid_loss'] >= config['valid_loss_early_stopping']:
                logging.info('Valid loss did not improve for {} epochs!'.format(config['valid_loss_early_stopping']))
                logging.info('[Valid loss early stopping]')
                break

        save_checkpoint(best_state, f'{experiment_dir}/best_state.ckpt')
        if args.save_best_valid_predictions:
            best_valid_predictions_df.to_csv(f'{experiment_dir}/valid_predictions_iteration_{iteration:04}.csv', index=False)

        refinement_stats['avg_mAP_over_iterations'].append(refinement_stats['this_iteration_avg_mAP'])
        logging.info(refinement_stats)

        if refinement_stats['this_iteration_avg_mAP'] >= np.max(refinement_stats['avg_mAP_over_iterations']):
            logging.info(f'Got a new best refinement model with avg_mAP = {refinement_stats["this_iteration_avg_mAP"]:.2f}')
            save_checkpoint(best_state, f'{experiment_dir}/best_refinement_state.ckpt')

        best_state, model, optimizer, scheduler, train_data_loader, valid_data_loader, train_pseudo_gt_stats, valid_pseudo_gt_stats = load_best_model_and_update_pseudo_gt(best_state, config, model, optimizer,scheduler, train_data_loader, valid_data_loader, train_detection_eval, valid_detection_eval, device)
        epoch_offset += config['max_epoch']
        dump_refinement_state_to_tensorboard(summary_writer, refinement_stats, train_pseudo_gt_stats, valid_pseudo_gt_stats, epoch_offset)

        # refinement early stopping
        if iteration - np.argmax(refinement_stats['avg_mAP_over_iterations']) >= config['refinement_early_stopping']:
            logging.info(f'Refinement did not improve avg mAP for {config["refinement_early_stopping"]} iterations!')
            logging.info(f'[Refinement early stopping]')
            break

    dump_best_model_metrics_to_tensorboard(summary_writer, best_state, refinement_stats)

    return best_state

def eval_model(args, config, model, best_model):
    model.load_state_dict(best_model, strict=False)

    logging.info(f'Seeding with seed {config["seed"]}')
    valid_detection_eval = EvalActionDetection(args.valid_labels_filename, tiou_thresholds=config['tiou_thresholds'],
                                               verbose=False)
    config, datasets, data_loaders = create_datasets(config, args)
    valid_metrics, valid_predictions_df = full_epoch(config, model, data_loaders['valid'], args.device,
                                                     valid_detection_eval)
    valid_predictions_df.to_csv(f'{os.path.dirname(args.checkpoint)}/final_valid_predictions.csv', index=False)
    print('This model mAP: {}'.format(valid_metrics['avg_mAP']))

def main(args):
    start_time = time.time()
    device = torch.device(args.device)
    logging.info(f'using device {args.device}')
    experiment_dir = generate_experiment_dir(args.log_dir)
    summary_writer = SummaryWriter(log_dir=experiment_dir)

    logging.info(f'logs and checkpoint will be saved in {experiment_dir}')

    if args.checkpoint is None:
        # train from scratch
        config = Config(args.config_type).create_config(alpha_pseudo_gt_loss=args.alpha_pseudo_gt_loss,
                                                        pseudo_gt_generator_type=args.pseudo_gt_generator_type,
                                                        pseudo_gt_dropout=args.pseudo_gt_dropout)
        config = backward_compatible_config(config)

        logging.info(f'Seeding with seed {config["seed"]}')
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])

        config, datasets, data_loaders = create_datasets(config, args)
        model = Model(num_classes=config['num_classes'], input_size=config['input_size'], num_layers=config['num_layers'],
                      alpha_l1_loss=config['alpha_l1_loss'], alpha_group_sparsity_loss=config['alpha_group_sparsity_loss'],
                      alpha_pseudo_gt_loss=config['alpha_pseudo_gt_loss'], weighted_psgt_loss=config['weighted_psgt_loss'],
                      pseudo_gt_loss_dim=config['pseudo_gt_loss_dim'])

        optimizer = Adam(model.parameters(), lr=config['initial_lr'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config['lr_decay'], patience=config['lr_patience'], verbose=True)
        initial_epoch = 0
        best_state = train(args, config, model, data_loaders['train'], data_loaders['valid'], device, optimizer,
                           scheduler, initial_epoch, experiment_dir, summary_writer)
    else:
        best_model = torch.load(args.checkpoint, map_location=args.device)
        config = backward_compatible_config(best_model['config'])
        model = Model(num_classes=config['num_classes'], input_size=config['input_size'],
                      num_layers=config['num_layers'],
                      alpha_l1_loss=config['alpha_l1_loss'],
                      alpha_group_sparsity_loss=config['alpha_group_sparsity_loss'],
                      alpha_pseudo_gt_loss=config['alpha_pseudo_gt_loss'],
                      weighted_psgt_loss=config['weighted_psgt_loss'],
                      pseudo_gt_loss_dim=config['pseudo_gt_loss_dim'])
        eval_model(args, config, model, best_model['model'])

    logging.info(f'DONE!! Total time {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Training a Mantis classification model',
                        formatter_class=ArgumentDefaultsHelpFormatter)
    # required arguments.
    parser.add_argument('--train_features_filename', required=True, type=str,
                      help='Path to an HDF5 file containing the train video features')
    parser.add_argument('--train_labels_filename', required=True, type=str,
                      help='Path to CSV file with the train ground truth labels')
    parser.add_argument('--valid_features_filename', required=True, type=str,
                      help='Path to an HDF5 file containing the test video features')
    parser.add_argument('--valid_labels_filename', required=True, type=str,
                      help='Path to CSV file with the test ground truth labels')
    parser.add_argument('--log_dir', required=True, type=str,
                      help='Where logs and model checkpoints are to be saved.')

    # optional arguments
    parser.add_argument('--device', default='cuda:0', type=str,
                      help='The GPU device to train on')
    parser.add_argument('--config_type', default='random', type=str,
                      help='The hyperparameter config type')
    parser.add_argument('--pseudo_gt_dropout', default=4, type=int,
                      help='pseudo_gt_dropout in percentage up to 100')
    parser.add_argument('--alpha_pseudo_gt_loss', default=4, type=int,
                      help='alpha_pseudo_gt_loss')
    parser.add_argument('--pseudo_gt_generator_type', default=1, type=int,
                      help='pseudo_gt_generator_type  0: oracle; 1: from predictions; 2: from attention; 3: from cas; 4: biased random 60/40; 5: uniform random')
    parser.add_argument('--checkpoint', default=None, type=str,
                      help='Path to the .ckpt file containing the checkpoint. If not given, then the train a model from scratch. If given, then load the model and evaluate.')
    parser.add_argument('--save_best_valid_predictions', action='store_true')
    parser.add_argument('--loglevel', default='INFO', type=str, help='logging level')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                    level=numeric_level)
    delattr(args, 'loglevel')

    main(args)
