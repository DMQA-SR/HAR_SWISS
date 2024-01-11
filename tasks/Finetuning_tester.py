import os
import torch

from model.Signal_Token_transformer import STT
from model.classifier import MLPHead
from sklearn.metrics import balanced_accuracy_score, f1_score


class HAR_tester(object):
    def __init__(self, args, test_loader, save_path, best=False):

        self.args = args
        self.best = best
        self.args = args
        
        if best:
            self.tag = 'best'
        else:
            self.tag = 'pt'
            
        # Define save directory
        self.check_path = os.path.join(save_path, f'ckpt')
        
        # Define Loader
        self.test_loader = test_loader

        # Define model
        encoder = STT(args)
        classifier = MLPHead(args, input_dim=args.emb_dim*args.num_signals, output_dim=args.num_classes)

        enc_weight = torch.load(os.path.join(self.check_path, f'encoder.{self.tag}'))
        classifier_weight = torch.load(os.path.join(self.check_path,f'classifier.{self.tag}'))
        
        encoder.load_state_dict(enc_weight)
        classifier.load_state_dict(classifier_weight)
        
        self.encoder = encoder.to(args.cuda)
        self.classifier = classifier.to(args.cuda)

    @torch.no_grad()
    def test(self):
        self.encoder.eval()
        self.classifier.eval()

        y_true_list, y_pred_list = [], []
        for i, (crt_x, crt_y) in enumerate(self.test_loader):
            """
            x : (batch, # of sensors, feature dim)
            y : (batch,)"""

            x = crt_x.to(device=self.args.cuda)
            y = crt_y.to(device=self.args.cuda, dtype=torch.int64)

            # features : {'enc_out':enc_out, 'mean_out':mean_out}
            features = self.encoder(x)
            dec_input = features['enc_out'].view(x.shape[0], -1) # shape : batch, signals*emb_dim

            pred = self.classifier(dec_input)

            # Calculate metric
            y_pred = torch.argmax(pred, dim=-1).detach().cpu().numpy() 
            y_true = y.detach().cpu().numpy()
            
            y_true_list += list(y_true)
            y_pred_list += list(y_pred)

        test_acc, test_f1 = self.matrics(y_true_list, y_pred_list)

        print(f'\nK_Fold : {self.args.exp_fold} | Dataset : {self.args.dataset}')
        print(f'Tag : {self.tag}')
        print(f'Test Accuracy     : {test_acc:2.4f} | \tTest F1-Score     : {test_f1:2.4f}')

    def matrics(self, y_true, y_pred):
        acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        return acc, f1