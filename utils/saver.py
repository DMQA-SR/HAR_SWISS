import os
import torch

from warnings import warn

class Saver:
    def __init__(self, path):
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

            warn(f'{path} does not exist. Creating.')

    def checkpoint(self, tag, payload, is_best=False):
        checkpoint_path = self.get_path(tag, is_best)

        with open(checkpoint_path, "wb+") as fp:
            _payload = payload.state_dict()
            torch.save(_payload, fp)

    def get_path(self, tag, is_best=False):
        if is_best:
            fname = f'{tag}.best'
        else:
            fname = f'{tag}.pt'
        checkpoint_path = os.path.join(self.path, fname)

        return checkpoint_path

    def load(self, tag, model, is_best=False):
        checkpoint_path = self.get_path(tag, is_best)
        
        if os.path.exists(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            _payload = model.state_dict()
            _payload.update(payload)

            model.load_state_dict(_payload)
            print('All keys matched successfully...')

        else:
            warn(f'Error: {checkpoint_path} No Weights loaded')
