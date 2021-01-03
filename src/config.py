import numpy as np
import logging
import torch
from tensorboardX import SummaryWriter

class Config(object):
    def __init__(self, config_type):
        switcher = {'random_untrimmednet_thumos14': self._create_random_untrimmednet_thumos14_config,
                    'random_untrimmednet_activitynet_v1-2': self._create_random_untrimmednet_activitynet_v1_2_config,
                    'random_i3d_thumos14': self._create_random_i3d_thumos14_config,
                    'random_i3d_activitynet_v1-2': self._create_random_i3d_activitynet_v1_2_config,
                    'beta_generator_study_untrimmednet_thumos14': self._create_beta_generator_study_untrimmednet_thumos14_config,
                    'beta_generator_study_i3d_thumos14': self._create_beta_generator_study_i3d_thumos14_config,
                    'beta_generator_study_untrimmednet_activitynet_v1-2': self._create_beta_generator_study_untrimmednet_activitynet_v1_2_config,
                    'beta_generator_study_i3d_activitynet_v1-2': self._create_beta_generator_study_i3d_activitynet_v1_2_config,
                    'dropout_study_untrimmednet_thumos14': self._create_dropout_study_untrimmednet_thumos14_config,
                    'dropout_study_i3d_thumos14': self._create_dropout_study_i3d_thumos14_config,
                    'dropout_study_untrimmednet_activitynet_v1-2': self._create_dropout_study_untrimmednet_activitynet_v1_2_config,
                    'dropout_study_i3d_activitynet_v1-2': self._create_dropout_study_i3d_activitynet_v1_2_config,
                    'best_thumos14': self._create_best_thumos14_config,
                    'best_activitynet_v1-2': self._create_best_activitynet_v1_2_config,
                    'hacs': self._create_best_hacs_config,
                    'random_hacs': self._create_random_hacs_config,
                    'playground': self._create_playground_config,
                    'playground_thumos': self._create_playground_thumos_config,
                    'playground_thumos_unt': self._create_playground_thumos_unt_config,
                    }
        self.create_config = switcher.get(config_type, None)

        logging.info('Setting config type to {}'.format(config_type))
        if self.create_config is None:
            logging.warn('Invalid config type {}. Setting config type is to the default playground config.'.format(config_type))
            self.create_config = self._create_playground_config

    ############################################################################################################
    ################################################# THUMOS14 #################################################
    ############################################################################################################
    def _create_random_untrimmednet_thumos14_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a random UntrimmedNet THUMOS14 config')
        config = {'num_workers': 4,
                  'batch_size': 2**5,
                  'snippet_size': 15,
                  'stream': 0, # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 2**rnd_choice(0,4,1,output_type=int),
                  'pseudo_gt_loss_dim': 1,  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': rnd_choice(0,1,1,output_type=int),
                  'alpha_fg_loss': 0,
                  'pseudo_gt_dropout': rnd_choice(0,0.8,0.2,output_type=float),

                  'num_layers': rnd_choice(1,3,1,output_type=int),

                  'top_k_labels': rnd_choice(1,3,1,output_type=int),
                  'min_cas_score': rnd_choice(0.01,0.15,0.01,output_type=float),
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels':  rnd_choice(1,3,1,output_type=int),
                  'refinement_min_cas_score': rnd_choice(0.1,0.6,0.02,output_type=float),
                  'refinement_min_attention_score': rnd_choice(0.92,0.99,0.01,output_type=float),
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': rnd_choice(1,3,1,output_type=int),

                  'pseudo_gt_generator_type': 1, # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random

                  'initial_lr': 10**rnd_choice(-4,-3,-1,output_type=float),
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': np.random.randint(424242),
                  'version_name': 'random_untrimmednet_thumos14',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                 }
        return config

    def _create_random_i3d_thumos14_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a random I3D THUMOS14 config')
        config = {'num_workers': 4,
                  'batch_size': 2**5,
                  'snippet_size': 16,
                  'stream': 1, # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 2**rnd_choice(0,4,1,output_type=int),
                  'pseudo_gt_loss_dim': 1,  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': rnd_choice(0,1,1,output_type=int),
                  'alpha_fg_loss': 0,
                  'pseudo_gt_dropout': rnd_choice(0,0.8,0.2,output_type=float),

                  'num_layers': rnd_choice(1,3,1,output_type=int),

                  'top_k_labels': rnd_choice(1,3,1,output_type=int),
                  'min_cas_score': rnd_choice(0.01,0.15,0.01,output_type=float),
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels':  rnd_choice(1,3,1,output_type=int),
                  'refinement_min_cas_score': rnd_choice(0.1,0.6,0.02,output_type=float),
                  'refinement_min_attention_score': rnd_choice(0.92,0.99,0.01,output_type=float),
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': rnd_choice(1,3,1,output_type=int),

                  'pseudo_gt_generator_type': 1, # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random

                  'initial_lr': 10**rnd_choice(-4,-3,-1,output_type=float),
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': np.random.randint(424242),
                  'version_name': 'random_untrimmednet_thumos14',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                 }
        return config

    def _create_beta_generator_study_untrimmednet_thumos14_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a beta_generator_study_untrimmednet_thumos14 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 5,
                  'snippet_size': 15,
                  'stream': 0,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0,
                  'alpha_pseudo_gt_loss': alpha_pseudo_gt_loss,
                  'alpha_bg_loss': 0.0,  # 1.0/20.0,
                  'pseudo_gt_loss_dim': 1,  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': 0.2,

                  'num_layers': 1,

                  'top_k_labels': 2,
                  'min_cas_score': 0.02,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels': 2,
                  'refinement_min_cas_score': 0.34,
                  'refinement_min_attention_score': 0.98,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': pseudo_gt_generator_type,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -3,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 329074,  # np.random.randint(424242),
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_dropout_study_untrimmednet_thumos14_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a dropout_study_untrimmednet_thumos14 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 5,
                  'snippet_size': 15,
                  'stream': 0,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0,
                  'alpha_pseudo_gt_loss': 4,
                  'alpha_bg_loss': 0.0,  # 1.0/20.0,
                  'pseudo_gt_loss_dim': 1,  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': pseudo_gt_dropout/100,

                  'num_layers': 1,

                  'top_k_labels': 2,
                  'min_cas_score': 0.02,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels': 2,
                  'refinement_min_cas_score': 0.34,
                  'refinement_min_attention_score': 0.98,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -3,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 329074,  # np.random.randint(424242),
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_beta_generator_study_i3d_thumos14_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a beta_generator_study_i3d_thumos14 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 5,
                  'snippet_size': 16,
                  'stream': 1,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0,
                  'alpha_pseudo_gt_loss': alpha_pseudo_gt_loss,
                  'alpha_bg_loss': 0.0,  # 1.0/20.0,
                  'pseudo_gt_loss_dim': 1,  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': 0.3,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.35,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels': 2,
                  'refinement_min_cas_score': 0.1,
                  'refinement_min_attention_score': 0.92,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': pseudo_gt_generator_type,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -3,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 106793,  # np.random.randint(424242),
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_dropout_study_i3d_thumos14_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a dropout_study_i3d_thumos14 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 5,
                  'snippet_size': 16,
                  'stream': 1,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0,
                  'alpha_pseudo_gt_loss': 2,
                  'alpha_bg_loss': 0.0,  # 1.0/20.0,
                  'pseudo_gt_loss_dim': 1,  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': pseudo_gt_dropout/100,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.35,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels': 2,
                  'refinement_min_cas_score': 0.1,
                  'refinement_min_attention_score': 0.92,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -3,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 5,
                  'reset': True,

                  'seed': 106793,  # np.random.randint(424242),
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_best_thumos14_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating the best I3D THUMOS14 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 5,
                  'snippet_size': 16,
                  'stream': 1,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0,
                  'alpha_pseudo_gt_loss': 2,
                  'alpha_bg_loss': 0.0,  # 1.0/20.0,
                  'pseudo_gt_loss_dim': 1,  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': 0.3,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.35,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels': 2,
                  'refinement_min_cas_score': 0.1,
                  'refinement_min_attention_score': 0.92,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -3,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 106793,  # np.random.randint(424242),
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    ############################################################################################################
    ############################################ ActivityNet v1.2 ##############################################
    ############################################################################################################
    def _create_random_untrimmednet_activitynet_v1_2_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a random UntrimmedNet ActivityNet v1.2 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 15,
                  'stream': 2, # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 2**rnd_choice(0,4,1,output_type=float),
                  'pseudo_gt_loss_dim': 2,  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 0,
                  'pseudo_gt_dropout': rnd_choice(0, 0.6, 0.2, output_type=float),

                  'num_layers': rnd_choice(1,3,1,output_type=int),

                  'top_k_labels': 2,
                  'min_cas_score':  rnd_choice(0.001, 0.01, 0.002, output_type=float),
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': rnd_choice(1,3,1,output_type=int),
                  'refinement_min_cas_score': rnd_choice(0.03,0.07,0.02,output_type=float),
                  'refinement_min_attention_score': rnd_choice(0.3,0.7,0.2,output_type=float),
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': rnd_choice(1,2,1,output_type=int),

                  'pseudo_gt_generator_type': 1, # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random

                  'initial_lr': 10**rnd_choice(-5,-4,1,output_type=float),
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 13,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': np.random.randint(424242),
                  'version_name': 'random_untrimmednet_activitynet_v1-2',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                 }
        return config

    def _create_random_i3d_activitynet_v1_2_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a random I3D ActivityNet v1.2 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 16,
                  'stream': 2,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 2 ** rnd_choice(0, 4, 1, output_type=float),
                  'pseudo_gt_loss_dim': 2, # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 0,
                  'pseudo_gt_dropout': rnd_choice(0, 0.6, 0.2, output_type=float),

                  'num_layers': rnd_choice(1,3,1,output_type=int),

                  'top_k_labels': 2,
                  'min_cas_score':  rnd_choice(0.005, 0.015, 0.005, output_type=float),
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': rnd_choice(1,3,1,output_type=int),
                  'refinement_min_cas_score': rnd_choice(0.03,0.09,0.02,output_type=float),
                  'refinement_min_attention_score': rnd_choice(0.3,0.7,0.2,output_type=float),
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': rnd_choice(1, 2, 1, output_type=int),

                  'pseudo_gt_generator_type': 1, # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random

                  'initial_lr': 10 ** rnd_choice(-5, -4, 1, output_type=float),
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 13,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': np.random.randint(424242),
                  'version_name': 'random_untrimmednet_activitynet_v1-2',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_beta_generator_study_untrimmednet_activitynet_v1_2_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a beta_generator_study_untrimmednet_activitynet_v1_2 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 15,
                  'stream': 2,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': alpha_pseudo_gt_loss,
                  'pseudo_gt_loss_dim': 2,
                  'weighted_psgt_loss': 0,
                  'pseudo_gt_dropout': 0.2,

                  'num_layers': 3,

                  'top_k_labels': 2,
                  'min_cas_score': 0.001,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': 3,
                  'refinement_min_cas_score': 0.07,
                  'refinement_min_attention_score': 0.5,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': pseudo_gt_generator_type,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 127239,
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_dropout_study_untrimmednet_activitynet_v1_2_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a dropout_study_untrimmednet_activitynet_v1_2 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 15,
                  'stream': 2,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 2,
                  'pseudo_gt_loss_dim': 2,
                  'weighted_psgt_loss': 0,
                  'pseudo_gt_dropout': pseudo_gt_dropout/100,

                  'num_layers': 3,

                  'top_k_labels': 2,
                  'min_cas_score': 0.001,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': 3,
                  'refinement_min_cas_score': 0.07,
                  'refinement_min_attention_score': 0.5,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 127239,
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_beta_generator_study_i3d_activitynet_v1_2_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a beta_generator_study_i3d_activitynet_v1_2 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 16,
                  'stream': 2,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': alpha_pseudo_gt_loss,
                  'pseudo_gt_loss_dim': 2,
                  'weighted_psgt_loss': 0,
                  'pseudo_gt_dropout': 0.2,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.005,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': 3,
                  'refinement_min_cas_score': 0.05,
                  'refinement_min_attention_score': 0.5,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': pseudo_gt_generator_type,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 237366,
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_dropout_study_i3d_activitynet_v1_2_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a dropout_study_i3d_activitynet_v1_2 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 16,
                  'stream': 2,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 4,
                  'pseudo_gt_loss_dim': 2,
                  'weighted_psgt_loss': 0,
                  'pseudo_gt_dropout': pseudo_gt_dropout/100,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.005,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': 3,
                  'refinement_min_cas_score': 0.05,
                  'refinement_min_attention_score': 0.5,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 237366,
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    def _create_best_activitynet_v1_2_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating the best I3D Activitynet v1.2 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 16,
                  'stream': 2,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 4,
                  'pseudo_gt_loss_dim': 2,
                  'weighted_psgt_loss': 0,
                  'pseudo_gt_dropout': 0.2,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.005,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': 3,
                  'refinement_min_cas_score': 0.05,
                  'refinement_min_attention_score': 0.5,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 237366,
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    ############################################################################################################
    ############################################ HACS ##############################################
    ############################################################################################################

    def _create_random_hacs_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a random UntrimmedNet ActivityNet v1.2 config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 1,
                  'stream': 2, # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 2**rnd_choice(0,4,1,output_type=float),
                  'pseudo_gt_loss_dim': rnd_choice(1,2,1,output_type=int),  # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': rnd_choice(0,1,1,output_type=int),
                  'pseudo_gt_dropout': rnd_choice(0, 0.6, 0.2, output_type=float),

                  'num_layers': rnd_choice(1,3,1,output_type=int),

                  'top_k_labels': 2,
                  'min_cas_score':  rnd_choice(0.001, 0.01, 0.002, output_type=float),
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': rnd_choice(1,3,1,output_type=int),
                  'refinement_min_cas_score': rnd_choice(0.03,0.07,0.02,output_type=float),
                  'refinement_min_attention_score': rnd_choice(0.3,0.7,0.2,output_type=float),
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': rnd_choice(1,2,1,output_type=int),

                  'pseudo_gt_generator_type': 1, # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random

                  'initial_lr': 10**rnd_choice(-5,-4,1,output_type=float),
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 13,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': np.random.randint(424242),
                  'version_name': 'random_hacs',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                 }
        return config

    def _create_best_hacs_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating the best HACS config')
        config = {'num_workers': 4,
                  'batch_size': 2 ** 8,
                  'snippet_size': 1,
                  'stream': 2,  # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 2,
                  'pseudo_gt_loss_dim': 2,
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': 0.2,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.001,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': 3,
                  'refinement_min_cas_score': 0.05,
                  'refinement_min_attention_score': 0.5,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1,
                  # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10 ** -4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 237366,
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                  }
        return config

    ############################################################################################################
    ############################################### Playground #################################################
    ############################################################################################################
    def _create_playground_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a playground config')
        config = {'num_workers': 4,
                  'batch_size': 2**8,
                  'snippet_size': 16,
                  'stream': 2, # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0.0,
                  'alpha_pseudo_gt_loss': 4,
                  'pseudo_gt_loss_dim': 2,
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': 0.2,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.005,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': np.linspace(0.5, 0.95, 10),

                  'refinement_top_k_labels': 3,
                  'refinement_min_cas_score': 0.05,
                  'refinement_min_attention_score': 0.5,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1, # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10**-4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'valid_loss_early_stopping': 50,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 3,
                  'reset': True,

                  'seed': 237366,
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                 }
        return config

    def _create_playground_thumos_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a playground config')
        config = {'num_workers': 4,
                  'batch_size': 2**5,
                  'snippet_size': 16,
                  'stream': 1, # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0,
                  'alpha_pseudo_gt_loss': 2,
                  'pseudo_gt_loss_dim': 1, # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': 0.3,

                  'num_layers': 2,

                  'top_k_labels': 2,
                  'min_cas_score': 0.35,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels': 2,
                  'refinement_min_cas_score': 0.1,
                  'refinement_min_attention_score': 0.92,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1, # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10**-3,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 5,
                  'reset': True,

                  'seed': 106793,#np.random.randint(424242),
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                 }
        return config

    def _create_playground_thumos_unt_config(self, alpha_pseudo_gt_loss, pseudo_gt_generator_type, pseudo_gt_dropout):
        logging.info('Generating a playground config')
        config = {'num_workers': 4,
                  'batch_size': 2**5,
                  'snippet_size': 15,
                  'stream': 0, # 0 for temporal only, 1 for spatial only, and 2 for both streams

                  'alpha_l1_loss': 0.0,
                  'alpha_group_sparsity_loss': 0,
                  'alpha_pseudo_gt_loss': 2,
                  'pseudo_gt_loss_dim': 1, # 2: Cross Entropy, 1: Logistic Regression
                  'weighted_psgt_loss': 1,
                  'pseudo_gt_dropout': 0.4,

                  'num_layers': 1,

                  'top_k_labels': 2,
                  'min_cas_score': 0.05,
                  'min_attention_score': 0.5,
                  'padding': 2,
                  'tiou_thresholds': [0.5],

                  'refinement_top_k_labels': 2,
                  'refinement_min_cas_score': 0.5,
                  'refinement_min_attention_score': 0.94,
                  'refinement_padding': 2,
                  'refinement_from_m_iterations': 1,

                  'pseudo_gt_generator_type': 1, # 0: oracle; 1: from predictions; 2: from attention; 3: from class activation; 4: biased random 60/40; 5: uniform random 6:gap

                  'initial_lr': 10**-4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 100,
                  'valid_loss_early_stopping': 100,
                  'refinement_iterations': 50,
                  'refinement_early_stopping': 5,
                  'reset': True,

                  'seed': np.random.randint(424242), #rm 231193
                  'version_name': 'random',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']
                 }
        return config

def rnd_choice(start, end, step=1, output_type=float):
    '''
    generates a random number in [start, end] with spacing 
    size equal to step. The value of end is included.
    '''
    nums = np.append(np.arange(start, end, step), end)
    return output_type(np.random.choice(nums))

def backward_compatible_config(config):
    if 'valid_loss_early_stopping' not in config.keys():
        config['valid_loss_early_stopping'] = config['max_epoch']
    if 'refinement_early_stopping' not in config.keys():
        config['refinement_early_stopping'] = 3
    if 'weighted_psgt_loss' not in config.keys():
        config['weighted_psgt_loss'] = False
    if 'pseudo_gt_loss_dim' not in config.keys():
        config['pseudo_gt_loss_dim'] = 2
    if 'pseudo_gt_dropout' not in config.keys():
        config['pseudo_gt_dropout'] = 0
    return config

def dump_config_details_to_tensorboard(summary_writer, config):
    for k, v in config.items():
        if k not in config['do_not_dump_in_tensorboard']:
            summary_writer.add_scalar('config/{}'.format(k), v, 0)
