import os
import sys
import numpy as np
import random
import logging
import subprocess

import soundfile as sf
import tqdm
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch import nn
import omegaconf
from read_emg import EMGDataset

from torch.utils.tensorboard import SummaryWriter
import constants as enc_constants
from utils import (SizeAwareSampler, collate_raw, init_voiced_datasets_emg_encoder_training, 
                   decollate_tensor, combine_fixed_length)
from align import align_from_distances
from main_constants import *


from emg_encoder import EMGEncoderTransformer, init_emg_encoder
from pathlib import Path
import yaml
from typing import Optional


def get_configs_of(config_dir):
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    
    merged_config = {}
    merged_config.update(preprocess_config)
    merged_config.update(model_config)  
    merged_config.update(train_config)
    
    return preprocess_config, model_config, train_config, merged_config


def test(model: EMGEncoderTransformer, testset, device, acoustic_model: Optional[nn.Module] = None):
    model.eval()
    if acoustic_model is not None:
        acoustic_model.eval()
        
    dataloader = torch.utils.data.DataLoader( testset, batch_size=1, collate_fn=collate_raw)
    
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(PHONEME_INVENTORY), 
                                  len(PHONEME_INVENTORY)))
    seq_len = enc_constants.SEQ_LEN 
    
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation'):
            emg_input = combine_fixed_length([t.to(device, non_blocking=True) for t in batch[DataType.REAL_EMG]], seq_len * 8)
            paired_input = combine_fixed_length([t.to(device, non_blocking=True) for t in batch[DataType.PAIRED_EMG]], seq_len * 8)

            pred, phoneme_pred = model(emg_input)
            paired_pred, paired_phoneme_pred = model(paired_input)  
            
            loss, phon_acc = dtw_loss(pred, phoneme_pred, paired_pred, paired_phoneme_pred, batch, True, phoneme_confusion)
            
            
            losses.append(loss.item())
            accuracies.append(phon_acc)
            
            

    model.train()
    return np.mean(losses), np.mean(accuracies), phoneme_confusion




def save_output(model, datapoint, filename, device, audio_normalizer, vocoder):
    model.eval()
    with torch.no_grad():
        sess = datapoint['session_ids'].to(device=device).unsqueeze(0)
        X = datapoint['emg'].to(dtype=torch.float32, device=device).unsqueeze(0)
        X_raw = datapoint['raw_emg'].to(dtype=torch.float32, device=device).unsqueeze(0)

        pred, _ = model(X, X_raw, sess)
        y = pred.squeeze(0)

        y = audio_normalizer.inverse(y.cpu()).to(device)

        audio = vocoder(y).cpu().numpy()

    sf.write(filename, audio, 22050)

    model.train()

def get_aligned_prediction(model, datapoint, device, audio_normalizer):
    model.eval()
    with torch.no_grad():
        silent = datapoint['silent']
        sess = datapoint['session_ids'].to(device).unsqueeze(0)
        X = datapoint['emg'].to(device).unsqueeze(0)
        X_raw = datapoint['raw_emg'].to(device).unsqueeze(0)
        y = datapoint['parallel_voiced_audio_features' if silent else 'audio_features'].to(device).unsqueeze(0)

        pred, _ = model(X, X_raw, sess) # (1, seq, dim)

        if silent:
            costs = torch.cdist(pred, y).squeeze(0)
            alignment = align_from_distances(costs.T.detach().cpu().numpy())
            pred_aligned = pred.squeeze(0)[alignment]
        else:
            pred_aligned = pred.squeeze(0)

        pred_aligned = audio_normalizer.inverse(pred_aligned.cpu())

    model.train()
    return pred_aligned






def dtw_loss(predictions, phoneme_predictions, paired_predictions, paired_phoneme_predictions, example, phoneme_eval=False, phoneme_confusion=None):  # audio -> mfccs -> audio_features (target)
    device = predictions.device

    speech_unit_predictions_list = decollate_tensor(predictions, example['speech_unit_lengths']) # preditions: torch.Size([79, 100, 256]) -> [torch.Size([204, 256]), torch.Size([114, 256]), ..., torch.Size([823, 256])] == 22
    phoneme_predictions_list = decollate_tensor(phoneme_predictions, example['speech_unit_lengths']) # phoneme_predictions: torch.Size([79, 100, 48]) -> [torch.Size([204, 48]), torch.Size([114, 48]), ..., torch.Size([823, 48])] == 32

    paired_speech_unit_predictions_list = decollate_tensor(paired_predictions, example['paired_speech_unit_lengths']) 
    paired_phoneme_predictions_list = decollate_tensor(paired_phoneme_predictions, example['paired_speech_unit_lengths'])
    
    batch_size = len(example['speech_unit_lengths'])
    assert len(speech_unit_predictions_list) == batch_size

    correct_phones = 0
    weight_speech_unit_loss = enc_constants.LOSS_WEIGHT_SPEECH_UNITS
    weight_phoneme_loss = enc_constants.LOSS_WEIGHT_PHONEMES

    speech_unit_targets_list = [t.to(device, non_blocking=True) for t in example[DataType.SPEECH_UNITS]] # [torch.Size([204, 256]), torch.Size([114, 256]), ..., torch.Size([882, 256])] == 22
    assert len( speech_unit_targets_list) == batch_size, f"Speech unit target list is not batch size {len(speech_unit_targets_list)} vs. {batch_size})"
    total_num_phone_targets = 0
    

    losses = []
    su_loss_norm = float(enc_constants.SU_LOSS_NORM)
    
    for sample_idx in range(batch_size):
        speech_unit_pred = speech_unit_predictions_list[sample_idx]
        speech_unit_target = speech_unit_targets_list[sample_idx]

        phoneme_prediction = phoneme_predictions_list[sample_idx]
        phoneme_target = example[DataType.PHONEMES][sample_idx].to(device)
        is_silent = example["silent"][sample_idx]
        
        # paired
        paired_speech_unit_pred = paired_speech_unit_predictions_list[sample_idx]
        paired_phoneme_prediction = paired_phoneme_predictions_list[sample_idx]
        
        
        if not is_silent:
            assert speech_unit_target.size(0) == speech_unit_pred.size(0)
            speech_unit_dists = F.pairwise_distance(speech_unit_target, speech_unit_pred, p=su_loss_norm)
            speech_unit_loss = speech_unit_dists.mean()

            phoneme_loss = F.cross_entropy(phoneme_prediction, phoneme_target, reduction='mean')
            # Total loss
            loss = ( (weight_speech_unit_loss * speech_unit_loss) + (weight_phoneme_loss * phoneme_loss) )
            losses.append(loss)

            if phoneme_eval:
                pred_phone = phoneme_prediction.argmax(-1)
                correct_phones += (pred_phone == phoneme_target).sum().item()
                total_num_phone_targets += len(phoneme_target)

                for p, t in zip(pred_phone.tolist(), phoneme_target.tolist()):
                    phoneme_confusion[p, t] += 1
                    
                    
        else:
            ########################### Speech Unit loss #########################
            speech_unit_dists = torch.cdist(speech_unit_pred.unsqueeze(0), speech_unit_target.unsqueeze(0), p=su_loss_norm)  
            speech_unit_costs = speech_unit_dists.squeeze(0)
            
            pred_phone = F.log_softmax(phoneme_prediction, -1) # pred_phone.shape == torch.Size([823, 48])  48개 phoneme class에 대한 확률 분포 (log-prob)
            phone_lprobs = pred_phone[:, phoneme_target]
            
            speech_unit_with_phone_costs = weight_speech_unit_loss * speech_unit_costs + weight_phoneme_loss * -phone_lprobs  # torch.Size([823, 882])
            alignment = align_from_distances(speech_unit_with_phone_costs.T.cpu().detach().numpy()) # len(alignment) == 882

            loss = speech_unit_with_phone_costs[alignment, range(len(alignment))].sum() / len(speech_unit_target)
            
            
            
            ################################# latent DTW ##########################
            paired_voiced_dists = torch.cdist(speech_unit_pred.unsqueeze(0), paired_speech_unit_pred.unsqueeze(0).detach(), p=su_loss_norm)  
            paired_voiced_costs = paired_voiced_dists.squeeze(0)
   

            # pred_phone (seq1_len, 48), phoneme_target (seq2_len) ->  phone_probs (seq1_len, seq2_len)
            paired_phoneme_dists = torch.cdist(phoneme_prediction.unsqueeze(0), paired_phoneme_prediction.unsqueeze(0).detach(), p=su_loss_norm)
            paired_phoneme_costs = paired_phoneme_dists.squeeze(0)
            
            
            paired_latent_with_phone_costs = weight_speech_unit_loss * paired_voiced_costs + weight_phoneme_loss * paired_phoneme_costs
            latent_alignment = align_from_distances(paired_latent_with_phone_costs.T.cpu().detach().numpy()) # len(alignment) == 882

            latent_loss = paired_latent_with_phone_costs[latent_alignment, range(len(latent_alignment))].sum() / len(paired_speech_unit_pred)
            
            
            alpha = 1.0
            beta = 1.0
            silent_loss = alpha * loss + beta * latent_loss   # silent's losses weight scale
            losses.append(silent_loss)
            


            if phoneme_eval: 
                alignment = align_from_distances(speech_unit_with_phone_costs.T.cpu().detach().numpy())

                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == phoneme_target).sum().item()
                total_num_phone_targets += len(phoneme_target)

                for p, t in zip(pred_phone[alignment].tolist(), phoneme_target.tolist()):
                    phoneme_confusion[p, t] += 1
        
    batch_loss = sum(losses) / batch_size
    if phoneme_eval:
        phone_acc = correct_phones / total_num_phone_targets
    else:
        phone_acc = float("nan")

    return batch_loss, phone_acc





def train_model(model_config, preprocess_config,  trainset: EMGDataset, devset: EMGDataset, device, output_directory: Path):
    n_epochs = 200
    training_subset = trainset
    

    dataloader = torch.utils.data.DataLoader(training_subset, pin_memory=(device=='cuda'), collate_fn=collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(training_subset, enc_constants.TRAIN_BATCH_MAX_LEN))

    model = init_emg_encoder(model_config, device)
    
    optim = torch.optim.AdamW(list(model.parameters()), weight_decay=enc_constants.WEIGHT_DECAY,)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=enc_constants.LEARNING_RATE_PATIENCE)
    
    
    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = enc_constants.LEARNING_RATE
    
    
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= enc_constants.LEARNING_RATE_WARMUP:
            set_lr(iteration * target_lr / enc_constants.LEARNING_RATE_WARMUP)

    seq_len = enc_constants.SEQ_LEN  # 200
    
    
    
    best_val_loss = float("inf")
    best_val_accs = 0.0
    batch_idx = 0
    num_no_improvement = 0
    
    # Mixed Precision Scaler    
    scaler = torch.cuda.amp.GradScaler()
    
    writer = SummaryWriter(  os.path.join(str(output_directory.absolute()), "logs")   )
    
    checkpoint_dir = os.path.join(str(output_directory), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch_idx in range(n_epochs):
        losses = []
        
        for batch in tqdm.tqdm(dataloader, 'Train step'):
            optim.zero_grad()
            schedule_lr(batch_idx)

            emg_input = combine_fixed_length([t.to(device, non_blocking=True) for t in batch[DataType.REAL_EMG]], seq_len * 8) # torch.Size([79, 1600, 8])
            paired_input = combine_fixed_length([t.to(device, non_blocking=True) for t in batch[DataType.PAIRED_EMG]], seq_len * 8)


            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred, phoneme_pred = model(emg_input)  # pred == torch.Size([79, 100, 256]), phoneme_pred == torch.Size([79, 100, 48])
                paired_pred, paired_phoneme_pred = model(paired_input)
                loss, phon_acc  = dtw_loss(pred, phoneme_pred, paired_pred, paired_phoneme_pred, batch)
                train_loss_item = loss.item()
                losses.append(train_loss_item)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            batch_idx += 1
            writer.add_scalar("train/loss", train_loss_item, global_step)
            writer.add_scalar("train/phon_acc", phon_acc, global_step)
            global_step += 1
            
            
        train_loss = np.mean(losses)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            val, phoneme_acc, _ = test(model, devset, device)
        writer.add_scalar("val/loss", val, global_step)
        writer.add_scalar("val/phon_acc", phoneme_acc, global_step)
        
        lr_sched.step(val)
        logging.info( f'finished epoch {epoch_idx + 1} - training loss: {train_loss:.4f} | Train Phoneme Acc: {phon_acc:.4f} |  validation loss: {val:.4f}  |  val. phoneme accuracy: {phoneme_acc * 100:.2f}')
        
        
        if (epoch_idx + 1) % 10 == 0:
            
            model_filename = f'epoch_{epoch_idx + 1}_model.pt'
            model_path = os.path.join(str(checkpoint_dir), model_filename)
            torch.save(model.state_dict(), model_path)

        
        
    return model




def main(configs: omegaconf.DictConfig, base_output_dir: Path, seed: int):
    
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
        
    preprocess_config, model_config, train_config, merged_config = configs


    output_directory = base_output_dir / f"seed_{seed}"
    os.makedirs(output_directory, exist_ok=True)
    
    
    # GPU Configuration
    gpu_id = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    
    
    
    
    logging.getLogger().setLevel(logging.INFO)
    log_file = output_directory / "log.txt"
    fh = logging.FileHandler(str(log_file.absolute()))
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    logging.info(f"+++ STARTING TRAINING +++")
    logging.info(sys.argv)




    # Save configuration
    config_file = output_directory / "config.yaml"
    logging.info(f"Saving configuration file under: {config_file}")
    if not config_file.exists():
        with open(config_file, '+w') as fp:
            OmegaConf.save(config=merged_config, f=fp.name)


    emg_dataset_root = Path("../data")  
    trainset, devset, _ = init_voiced_datasets_emg_encoder_training(emg_dataset_root)  
    logging.info('train / dev split: %d %d',len(trainset),len(devset))



    logging.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    model = train_model(model_config, preprocess_config, trainset, devset, device, output_directory)

    logging.info(f"Finished training the model")
    
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_name", type=str, default="EMG_Encoder", help="name of model")
    parser.add_argument( "--config_path", type=str, default="../config")
    parser.add_argument("--exp_dir", type=Path, default=Path("../result"))
    args = parser.parse_args()

    
    # Read Config
    config_dir = os.path.join(args.config_path, args.model_name)
    configs = get_configs_of(config_dir)
    
    # Make base_output_dir
    os.makedirs(args.exp_dir , exist_ok=True)
    base_output_dir = Path(os.path.join(args.exp_dir, "EMGTransformer-DTW"))
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Loop over seeds
    seed_list = [0]
    
    for seed in seed_list:
        main( configs, base_output_dir, seed)
