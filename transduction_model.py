import os
import sys
import numpy as np
import logging
import subprocess
import random
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from read_emg import EMGDataset, SizeAwareSampler
from architecture_DTW import Model
from align import align_from_distances
from data_utils import phoneme_inventory, decollate_tensor, combine_fixed_length


from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'training batch size')
flags.DEFINE_integer('epochs', 200, 'number of training epochs')
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_integer('learning_rate_warmup', 500, 'steps of linear warmup')
flags.DEFINE_string('start_training_from', None, 'start training from this model')
flags.DEFINE_float('data_size_fraction', 1.0, 'fraction of training data to use')
flags.DEFINE_float('phoneme_loss_weight', 0.5, 'weight of auxiliary phoneme prediction loss')
flags.DEFINE_float('l2', 1e-5, 'weight decay')
flags.DEFINE_float('su_loss_norm', 2.0, 'su_loss_norm')
flags.DEFINE_string('output_directory', '../exp_results', 'output directory')

def test(model, testset, device):
    model.eval()

    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, collate_fn=testset.collate_raw)
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(phoneme_inventory),len(phoneme_inventory)))
    seq_len = 200
    

    

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, 'Validation', disable=None):
            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len*8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)
            paired_input = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['paired_raw_emg']], seq_len*8)
            
            
            pred, phoneme_pred, hidden_states = model(X, X_raw, sess)
            paired_pred, paired_phoneme_pred, paired_hidden_states = model(X, paired_input, sess)
            
            
            loss, phon_acc = dtw_loss(pred, phoneme_pred, paired_pred, paired_phoneme_pred, hidden_states, paired_hidden_states, batch, phoneme_eval=True, phoneme_confusion=phoneme_confusion)
            losses.append(loss.item())
            accuracies.append(phon_acc)


    model.train()
    return np.mean(losses), np.mean(accuracies), phoneme_confusion




def dtw_loss(predictions, phoneme_predictions, paired_predictions, paired_phoneme_predictions, hidden_states, paired_hidden_states, example, phoneme_eval=False, phoneme_confusion=None):  
    device = predictions.device

    predictions = decollate_tensor(predictions, example['lengths']) 
    phoneme_predictions = decollate_tensor(phoneme_predictions, example['lengths']) 

    audio_features = [t.to(device, non_blocking=True) for t in example['audio_features']] 

    phoneme_targets = example['phonemes']

    hidden_states_layers = [ decollate_tensor(h.contiguous(), example['lengths']) for h in hidden_states ] 
    paired_hidden_states_layers = [ decollate_tensor(h.contiguous(), example['paired_lengths']) for h in paired_hidden_states ]
    
    paired_mel_predictions_list = decollate_tensor(paired_predictions, example['paired_lengths']) 
    paired_phoneme_predictions_list = decollate_tensor(paired_phoneme_predictions, example['paired_lengths'])
    
    
    hidden_states_by_sample = list(zip(*hidden_states_layers))
    paired_hidden_states_by_sample = list(zip(*paired_hidden_states_layers))

    batch_size = len(example['lengths'])
    assert len(predictions) == batch_size
    
    total_num_phone_targets = 0
    

    losses = []
    layer_ids = range(len(hidden_states))
    correct_phones = 0

    for pred, y, phoneme_prediction, y_phone, silent, paired_mel_pred, paired_phoneme_prediction, hidden_states, paired_hidden_states in zip(predictions, audio_features, phoneme_predictions, phoneme_targets, example['silent'], paired_mel_predictions_list, paired_phoneme_predictions_list, hidden_states_by_sample, paired_hidden_states_by_sample):
        assert len(pred.size()) == 2 and len(y.size()) == 2
        y_phone = y_phone.to(device) 

        if silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0), p=FLAGS.su_loss_norm) 
            costs = dists.squeeze(0) 

          
            pred_phone = F.log_softmax(phoneme_prediction, -1) 
            phone_lprobs = pred_phone[:,y_phone]  

            costs = 0.5 * costs + FLAGS.phoneme_loss_weight * -phone_lprobs   

            alignment = align_from_distances(costs.T.cpu().detach().numpy()) 

            mel_loss = costs[alignment,range(len(alignment))].sum() / len(y)  
            
            ################################# latent DTW ##########################
            paired_voiced_dists = torch.cdist(pred.unsqueeze(0), paired_mel_pred.unsqueeze(0).detach(), p=FLAGS.su_loss_norm)  
            paired_voiced_costs = paired_voiced_dists.squeeze(0)
            
            paired_phoneme_dists = torch.cdist(phoneme_prediction.unsqueeze(0), paired_phoneme_prediction.unsqueeze(0).detach(), p=FLAGS.su_loss_norm)
            paired_phoneme_costs = paired_phoneme_dists.squeeze(0)
            
            
            paired_latent_with_phone_costs = 0.5 * paired_voiced_costs + FLAGS.phoneme_loss_weight * paired_phoneme_costs
            latent_alignment = align_from_distances(paired_latent_with_phone_costs.T.cpu().detach().numpy())
            
            latent_loss = paired_latent_with_phone_costs[latent_alignment, range(len(latent_alignment))].sum() / len(paired_mel_pred)
            
            
            ################################# Transformer layer DTW ######################################
            layer_losses = []
            for l in layer_ids:
                h_i  = hidden_states[l]         
                ph_i = paired_hidden_states[l].detach()  

                h_i_norm = F.normalize(h_i, p=2, dim=-1)   
                ph_i_norm = F.normalize(ph_i, p=2, dim=-1)
                
                cos_sim_matrix = torch.mm(h_i_norm, ph_i_norm.t())
                

                matched_cos_sim = cos_sim_matrix[latent_alignment, range(len(latent_alignment))]
                loss_h = (1 - matched_cos_sim).sum() / len(ph_i)
                layer_losses.append(loss_h)

            hidden_latent_loss = torch.stack(layer_losses).mean() 
            

            with torch.no_grad():   

                threshold = 8.0 
                diff = 1.0 * (threshold - latent_loss.detach())
                sample_weight = torch.sigmoid(diff)

                        
            weighted_latent_loss = latent_loss * sample_weight
            weighted_hidden_latent_loss = hidden_latent_loss * sample_weight
            
            alpha, beta, gamma = 1.0, 1.0, 1.0
           
            loss = alpha * mel_loss + beta * weighted_latent_loss + gamma * weighted_hidden_latent_loss # silent's losses weight scale
            losses.append(loss)
            

            if phoneme_eval: 
                alignment = align_from_distances(costs.T.cpu().detach().numpy())

                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == y_phone).sum().item()
                total_num_phone_targets += len(y_phone)
                
                for p, t in zip(pred_phone[alignment].tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1
                    
                    
                    
        else:
            assert y.size(0) == pred.size(0)
            dists = F.pairwise_distance(y, pred, p=FLAGS.su_loss_norm)
            mel_loss = dists.mean()
            
            
            phoneme_loss = F.cross_entropy(phoneme_prediction, y_phone, reduction='mean')
            loss = ((0.5 * mel_loss) + (FLAGS.phoneme_loss_weight * phoneme_loss))
            losses.append(loss)
            
            
            if phoneme_eval:
                pred_phone = phoneme_prediction.argmax(-1)
                correct_phones += (pred_phone == y_phone).sum().item()
                total_num_phone_targets += len(y_phone)
                
                for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1


    batch_loss = sum(losses) / batch_size
    if phoneme_eval:
        phone_acc = correct_phones / total_num_phone_targets
    else:
        phone_acc = float("nan")
        
    return batch_loss, phone_acc


def train_model(trainset, devset, device, output_dir):
    n_epochs = FLAGS.epochs

    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = trainset.subset(FLAGS.data_size_fraction)
    dataloader = torch.utils.data.DataLoader(training_subset, pin_memory=(device=='cuda'), collate_fn=devset.collate_raw, num_workers=0, batch_sampler=SizeAwareSampler(training_subset, 128000))

    n_phones = len(phoneme_inventory)
    
    emgencoder_ckpt_path = "../emg_encoder.pt"
    model = Model(devset.num_features, devset.num_speech_features, n_phones).to(device)
    state_dict = torch.load(emgencoder_ckpt_path, map_location={"device": device})
    model.load_state_dict(state_dict, strict=False)
    model.train()
    


    scaler = torch.cuda.amp.GradScaler()
    optim = torch.optim.AdamW(model.parameters(), weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=FLAGS.learning_rate_patience)


    
    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)

    writer = SummaryWriter(  os.path.join(output_dir, "logs")   )
    
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    seq_len = 200
    

    batch_idx = 0
    global_step = 0
    for epoch_idx in range(n_epochs):
        losses = []
        for batch in tqdm.tqdm(dataloader, 'Train step', disable=None):
            optim.zero_grad()
            schedule_lr(batch_idx)

            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['emg']], seq_len)  
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['raw_emg']], seq_len*8) 
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['session_ids']], seq_len)
            paired_input = combine_fixed_length([t.to(device, non_blocking=True) for t in batch['paired_raw_emg']], seq_len*8)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred, phoneme_pred, hidden_states = model(X, X_raw, sess) 
                paired_pred, paired_phoneme_pred, paired_hidden_states = model(X, paired_input, sess)
            
            
                loss, _ = dtw_loss(pred, phoneme_pred,  paired_pred, paired_phoneme_pred, hidden_states, paired_hidden_states, batch)
                losses.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            
            batch_idx += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1
            
            
        train_loss = np.mean(losses)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            val, phoneme_acc, _ = test(model, devset, device)
        writer.add_scalar("val/loss", val, global_step)
        writer.add_scalar("val/phon_acc", phoneme_acc, global_step)
        
                
        lr_sched.step(val)
        logging.info(f'finished epoch {epoch_idx+1} - validation loss: {val:.4f} training loss: {train_loss:.4f} phoneme accuracy: {phoneme_acc*100:.2f}')

        
        
        if (epoch_idx + 1) % 10 == 0:
            model_filename = f'epoch_{epoch_idx + 1}_model.pt' 
            model_path = os.path.join(str(checkpoint_dir), model_filename)
            torch.save(model.state_dict(), model_path)
         
    
    return model



def main(base_output_dir, seed):
    
    # Set output dir for this seed
    output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)

    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(output_dir, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    logging.info(subprocess.run(['git','rev-parse','HEAD'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(subprocess.run(['git','diff'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(sys.argv)
    
    
    
    # Set seed
    logging.info(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    


    trainset = EMGDataset(dev=False,test=False)  
    devset = EMGDataset(dev=True)                
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d',len(trainset),len(devset))

    
    # GPU Configuration
    gpu_id = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    model = train_model(trainset, devset, device, output_dir)



if __name__ == '__main__':
    FLAGS(sys.argv)
    
    # Make base_output_dir
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    
    # Loop over seeds
    seed_list = [0, 27, 42, 77, 1234, 1235, 1236, 1237, 1238, 1240]
    
    for seed in seed_list:
        main(FLAGS.output_directory, seed)
