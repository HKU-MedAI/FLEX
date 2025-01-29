import numpy as np
import torch
from utils.utils import *
import os
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from models.transmil import TransMIL
from models.model_dsmil import MILNet
from models.model_hierarchical_mil import HIPT_None_FC, HIPT_LGP_FC
from models.dtfdmil import DTFD_MIL
from models.abmil import DAttention
from models.catemil import CATEMIL
from models.catemil_pathgenclip import CATEMIL_PathGenClip
from models.ilra import ILRA
from models.acmil import ACMIL_GA
from models.catemil_quilt import CATEMIL_Quilt
import time

from torchsummary import summary

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, start_epoch=2, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.start_epoch = start_epoch

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss
        if epoch >= self.start_epoch:
            # return False
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
            elif score < self.best_score:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience or epoch > self.stop_epoch:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, ckpt_name)
                self.counter = 0
                if epoch > self.stop_epoch:
                    self.early_stop = True

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\n------------- Training Fold {}! -------------'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        # from tensorboardX import SummaryWriter
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    train_split, val_split, test_split, out_test_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    print("Out of distribution testing on {} samples".format(len(out_test_split)))

    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    # Init model
    if args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.B > 0:
        model_dict.update({'k_sample': args.B})
    
    if args.model_type == 'catemil':
        model = CATEMIL(n_classes=args.n_classes, task=args.task, fold=cur, exp_code=args.exp_code, n_ctx=args.len_learnable_prompt, base_mil=args.base_mil, slide_align=args.slide_align)
    else:
        raise NotImplementedError

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(device)
        
    optimizer = get_optim(model, args)

    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    out_test_loader = get_split_loader(out_test_split, testing = args.testing)

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=10, stop_epoch=20, verbose=True)

    else:
        early_stopping = None

    for epoch in range(args.max_epochs):
        print(f'\nProcessing Epoch {epoch} ...', end=' ')
        time_start = time.perf_counter()
        train_loop_hit(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, model_type=args.model_type, base_mil=args.base_mil, args=args)
        stop = validate_hit(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)
        
        time_end = time.perf_counter()
        print(f'Epoch time: {time_end - time_start}')
        
        if stop: 
            break
        
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        if hasattr(model, 'text_encoder'):
            delattr(model, 'text_encoder')
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    _, val_error, val_auc, _, val_f1 = summary_my(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger, test_f1 = summary_my(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    out_results_dict, out_test_error, out_test_auc, acc_logger, out_test_f1 = summary_my(model, out_test_loader, args.n_classes)

    acc_list = []
    correct_list = []
    count_list = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_list.append(acc)
        correct_list.append(correct)
        count_list.append(count)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
    print('Overall: acc {:.4f}, correct {}/{}'.format(np.sum(acc_list)/2, np.sum(correct_list), np.sum(count_list)))

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/val_f1', val_f1, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.add_scalar('final/test_f1', test_f1, 0)
        writer.close()
    
    return results_dict, out_results_dict, test_auc, val_auc, out_test_auc, 1-test_error, 1-val_error, 1-out_test_error, test_f1, val_f1, out_test_f1


def train_loop_hit(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None, model_type=None, base_mil=None, args=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_loss2 = 0.
    train_error = 0.
    train_inst_loss = 0.
    train_infonce_loss = 0.
    train_kl_loss = 0.
    train_op_loss = 0.
    train_cor_loss = 0.
    train_concept_loss = 0.
    train_mi = 0.
    train_loglikeli = 0.

    print('\n')
    for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
        data, data2, label = data.to(device, non_blocking=True), data2.to(device, non_blocking=True), label.to(device, non_blocking=True)
        data[data != data] = 0
        logits0, Y_hat, Y_prob, results_dict = model(data, data2, label, pretrain=pretrain)

        acc_logger.log(Y_hat, label)

        if 'infonce_loss' in results_dict.keys() and results_dict['infonce_loss'] is not None:
            infonce_loss = results_dict['infonce_loss']
            infonce_loss_value = infonce_loss.item()
            train_infonce_loss += infonce_loss_value

        if 'kl_loss' in results_dict.keys() and results_dict['kl_loss'] is not None:
            kl_loss = results_dict['kl_loss']
            kl_loss_value = kl_loss.item()
            train_kl_loss += kl_loss_value

        if 'op_loss' in results_dict.keys() and results_dict['op_loss'] is not None:
            op_loss = results_dict['op_loss']
            op_loss_value = op_loss.item()
            train_op_loss += op_loss_value

        if 'cor_loss' in results_dict.keys() and results_dict['cor_loss'] is not None:
            cor_loss = results_dict['cor_loss']
            cor_loss_value = cor_loss.item()
            train_cor_loss += cor_loss_value

        if 'concept_loss' in results_dict.keys() and results_dict['concept_loss'] is not None:
            concept_loss = results_dict['concept_loss']
            concept_loss_value = concept_loss.item()
            train_concept_loss += concept_loss_value

        loss = loss_fn(logits0, label)
        loss_value = loss.item()
        total_loss = loss + args.w_infonce * infonce_loss + args.w_kl * kl_loss
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(loader)
    train_loss2 /= len(loader)
    train_error /= len(loader)
    
    print(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_loss2:  {train_loss2:.4f}, train_error: {train_error:.4f}, train_infonce_loss: {train_infonce_loss:.4f}, train_kl_loss: {train_kl_loss:.4f}, train_op_loss: {train_op_loss:.4f}, train_mi: {train_mi:.4f}, train_loglikeli: {train_loglikeli:.4f}')
    acc_list = []
    correct_list = []
    count_list = []
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_list.append(acc)
        correct_list.append(correct)
        count_list.append(count)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    print('Overall: acc {:.4f}, correct {}/{}'.format(np.sum(acc_list)/2, np.sum(correct_list), np.sum(count_list)))

def validate_hit(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.eval()
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    # sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
            data, data2, label = data.to(device), data2.to(device), label.to(device)      
            data[data != data] = 0
            logits, Y_hat, Y_prob, _ = model(data, data2)
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            # import ipdb;ipdb.set_trace()
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    # if inst_count > 0:
    #     val_inst_loss /= inst_count
    #     for i in range(2):
    #         acc, correct, count = inst_logger.get_summary(i)
    #         print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    # if writer:
    #     writer.add_scalar('val/loss', val_loss, epoch)
    #     writer.add_scalar('val/auc', auc, epoch)
    #     writer.add_scalar('val/error', val_error, epoch)

    acc_list = []
    correct_list = []
    count_list = []
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        acc_list.append(acc)
        correct_list.append(correct)
        count_list.append(count)
        # import ipdb;ipdb.set_trace()
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
    print('Overall: acc {:.4f}, correct {}/{}'.format(np.sum(acc_list)/2, np.sum(correct_list), np.sum(count_list)))

    # wandb.log({"val/loss": val_loss, "val/auc": auc, "val/acc": 1-val_error})
     
    # import ipdb;ipdb.set_trace()
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary_my(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    # club = model[1]
    # model = model[0]
    # model.eval()
    # club.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
        data, data2, label = data.to(device), data2.to(device), label.to(device)
        data[data != data] = 0
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_hat, Y_prob, _ = model(data, data2)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        f1 = f1_score(all_labels, np.argmax(all_probs, axis=1))
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
        f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')


    return patient_results, test_error, auc, acc_logger, f1

def summary_clam(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, data2, label, slide_id, coords) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_hat, Y_prob, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        # aucs = []
        f1 = f1_score(all_labels, np.argmax(all_probs, axis=1))
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))
        f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='weighted')


    return patient_results, test_error, auc, acc_logger, f1