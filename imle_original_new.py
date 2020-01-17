'''
Code for Implicit Maximum Likelihood Estimation

This code implements the method described in the Implicit Maximum Likelihood
Estimation paper, which can be found at https://arxiv.org/abs/1809.09087

Copyright (C) 2018    Ke Li


This file is part of the Implicit Maximum Likelihood Estimation reference
implementation.

The Implicit Maximum Likelihood Estimation reference implementation is free
software: you can redistribute it and/or modify it under the terms of the GNU
Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

The Implicit Maximum Likelihood Estimation reference implementation is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the Implicit Maximum Likelihood Estimation reference implementation.
If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as vutils
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import random
import sys
import os
import os.path
import warnings
sys.path.append('./dci_code')
from dci import DCI
import collections
import argparse
from tqdm import tqdm
from collections import deque
import signal
import itertools

Hyperparams = collections.namedtuple('Hyperparams', 'optimizer base_lr momentum batch_size sample_db_size decay_step decay_rate staleness num_samples_factor')
Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None)

Config = collections.namedtuple('Config', 'num_epochs shuffle_data checkpoint_interval plot_interval plot_subinterval track_subinterval path_prefix device disable_cuda disable_multigpu num_threads seed')
Config.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None, None, None, None)

save_now = False
plot_now = False
def signal_handler_menu(sig, frame):
    global save_now
    global plot_now
    print("======== Script Control ========")
    print("Current script: %s" % (os.path.abspath(__file__)))
    while True:
        option = raw_input("(Q)uit, (S)ave, (P)lot or (C)ontinue? ")
        valid_option = False
        if "s" in option.lower():
            save_now = True
            valid_option = True
        if "p" in option.lower():
            plot_now = True
            valid_option = True
        if "c" in option.lower():
            valid_option = True
        if option.lower() == "q":
            raise KeyboardInterrupt
        if valid_option:
            break
        else:
            print("Invalid option.")

def signal_handler_noop(sig, frame):
    raise KeyboardInterrupt

class RandomDataset(data.Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class ZippedDataset(data.Dataset):

    def __init__(self, *datasets):
        assert all(len(datasets[0]) == len(dataset) for dataset in datasets)
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return len(self.datasets[0])

class ChoppedDataset(data.Dataset):

    def __init__(self, dataset, num_elems):
        self.dataset = dataset
        self.num_elems = num_elems

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(len(self.dataset), self.num_elems)

class SlicedDataset(data.Dataset):

    def __init__(self, dataset, slice_indices):
        self.dataset = dataset
        self.slices = slice_indices

    def __getitem__(self, index):
        return tuple(self.dataset[index][s] for s in self.slices)

    def __len__(self):
        return len(self.dataset)

class ConvImplicitModel(nn.Module):
    def __init__(self, z_dim):
        super(ConvImplicitModel, self).__init__()
        self.z_dim = z_dim
        self.tconv1 = nn.ConvTranspose2d(z_dim[0], 1024, z_dim[1], z_dim[2], bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, z):
        z = self.relu(self.bn1(self.tconv1(z)))
        z = self.relu(self.bn2(self.tconv2(z)))
        z = self.relu(self.bn3(self.tconv3(z)))
        z = torch.sigmoid(self.tconv4(z))
        return z


class CondConvImplicitModel(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(CondConvImplicitModel, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.tconv1 = nn.ConvTranspose2d(input_dim[0] + z_dim[0], 1024, z_dim[1], z_dim[2], bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, input_data, z):
        z = torch.cat((input_data, z), 1)
        z = self.relu(self.bn1(self.tconv1(z)))
        z = self.relu(self.bn2(self.tconv2(z)))
        z = self.relu(self.bn3(self.tconv3(z)))
        z = torch.sigmoid(self.tconv4(z))
        return z

class IMLE():
    # z_dim is a tuple, e.g.: (C, H, W)
    def __init__(self, z_dim, model, hyperparams, config, freeze_batch_norm = True):
        self.z_dim = z_dim
        self.model = model.to(device=config.device)
        if hasattr(self.model, "get_initializer"):
            self.model.apply(self.model.get_initializer())

        if hyperparams.optimizer == "adam":
        	self.optimizer = optim.Adam(self.model.parameters(), lr=hyperparams.base_lr, betas=hyperparams.momentum, weight_decay=0.0005)
        elif hyperparams.optimizer == "sgd":
        	self.optimizer = optim.SGD(self.model.parameters(), lr=hyperparams.base_lr, momentum=hyperparams.momentum, weight_decay=0.0005)
        else:
        	raise Exception("Unknown optimizer: optimizer must be either 'adam' or 'sgd'.")

        if not config.disable_cuda and torch.cuda.is_available():
            if not config.disable_multigpu and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.dci_db = None
        self.starting_epoch = 0
        self.init_z_data = None
        self.init_z_data_file = None
        self.testing_only = False		# If model weights are loaded but optimizer state is not loaded
        self.freeze_batch_norm = freeze_batch_norm
        self.tracked_context = dict()   # Populated automatically
        self.tracked_content = dict()   # Populated manually

    def set_lr(self, lr):
        # Set new learning rate
        for i, group in enumerate(self.optimizer.param_groups):
            print("Original learning rate for parameter group %d is %f; changed to %f and zeroed gradient statistics" % (i, group['lr'], lr))
            if lr != group['lr']:
                #print("Original learning rate for parameter group %d is %f; changed to %f and zeroed gradient statistics" % (i, group['lr'], lr))
                group['lr'] = lr
                for p in group['params']:
                    # Reset optimizer state
                    del self.optimizer.state[p]
                    self.optimizer.state[p] = dict()

    def _set_to_train_mode(self):
        self.model.train()

        if self.freeze_batch_norm:

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    #print("Setting %s layer to eval mode" % (classname))
                    m.eval()

            self.model.apply(set_bn_eval)

    def _set_to_eval_mode(self):
        self.model.eval()

    def _reset_optimizer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # Reset optimizer state
                if p in self.optimizer.state:
                    del self.optimizer.state[p]
                    self.optimizer.state[p] = dict()

    def _user_confirm(self):
        while True:
            option = raw_input("(Y)es or (N)o? ")
            if option.lower() == "y":
                break
            elif option.lower() == "n":
                raise KeyboardInterrupt
            else:
                print("Invalid option.")

    def _track(self, loc, event, cur_context, cur_content, config):
        self.tracked_context[loc] = cur_context
        context = dict()
        dirs = loc.split(":")
        for i in range(len(dirs)):
            cur_key = ":".join(dirs[:i+1])
            if cur_key in self.tracked_context:
                context.update(self.tracked_context[cur_key])

        content = dict()
        for i in range(len(dirs)):
            cur_key = ":".join(dirs[:i+1])
            if cur_key in self.tracked_content:
                content.update(self.tracked_content[cur_key])
        content.update(cur_content)

        lambda_content = dict()
        for key in content:
            if callable(content[key]):
                lambda_content[key] = content[key]
            else:
                lambda_content[key] = lambda key=key: content[key]

        self.track(loc, event, context, lambda_content, config)

        if event == "end":
            del self.tracked_context[loc]

    def track(self, loc, event, context, content, config):
        if loc == "epoch:samp" and event == "end":
            if loc not in self.tracked_content:
                self.tracked_content[loc] = dict()
            self.tracked_content[loc]["is_new"] = True
        elif loc.startswith("epoch:train"):
            if loc == "epoch:train" and event == "start":
                comb_dataset = content["comb_dataset"]()
                data_loader = data.DataLoader(comb_dataset, batch_size=7, shuffle=False, num_workers=1, pin_memory=False)
                (cur_batch_data,), (cur_batch_z,) = next(iter(data_loader))
                if loc not in self.tracked_content:
                    self.tracked_content[loc] = dict()
                self.tracked_content[loc]["z"]  = np.copy(cur_batch_z.cpu().data.numpy())
                self.tracked_content[loc]["data"] = np.copy(cur_batch_data.cpu().data.numpy())
                if "samples" not in self.tracked_content[loc]:
                    self.tracked_content[loc]["samples"] = deque(maxlen=8)
                    self.tracked_content[loc]["samples_info"] = deque(maxlen=8)
                self._set_to_eval_mode()
                cur_batch_samples = self.model(cur_batch_z.to(device=config.device)).cpu().data.numpy()
                self._set_to_train_mode()
                is_new = "epoch:samp" not in self.tracked_content or self.tracked_content["epoch:samp"]["is_new"]
                if is_new:
                    if "epoch:samp" not in self.tracked_content:
                         self.tracked_content["epoch:samp"] = dict()
                    self.tracked_content["epoch:samp"]["is_new"] = False
                    self.tracked_content[loc]["init_samples"] = cur_batch_samples
                self.tracked_content[loc]["samples"].append(cur_batch_samples)
                self.tracked_content[loc]["samples_info"].append({"epoch_start": True, "new": is_new})
            elif loc == "epoch:train:iter" and event == "prog":
                cur_batch_z = torch.from_numpy(content["z"]())
                self._set_to_eval_mode()
                cur_batch_samples = self.model(cur_batch_z.to(device=config.device)).cpu().data.numpy()
                self._set_to_train_mode()
                batch_idx = context["batch_idx"]
                if (batch_idx+1) % config.track_subinterval == 0:
                    self.tracked_content["epoch:train"]["samples"].append(cur_batch_samples)
                    self.tracked_content["epoch:train"]["samples_info"].append({"epoch_start": False, "new": False})
            elif loc == "epoch:train" and event == "end":
                del self.tracked_context["%s:iter" % (loc)]

    def plot(self, loc, event, context, config):
        def add_border(imgs, colour, border_width):
            imgs_proc = np.copy(imgs)
            assert(len(colour) == 3)
            for i in range(len(colour)):
                imgs_proc[:,i,:,:border_width] = colour[i]  # Left border
                imgs_proc[:,i,:,-border_width:] = colour[i] # Right border
                imgs_proc[:,i,:border_width,:] = colour[i]  # Top border
                imgs_proc[:,i,-border_width:,:] = colour[i] # Bottom border
            return imgs_proc

        if loc.startswith("epoch:train") and "epoch:train" in self.tracked_content:

            init_samples = self.tracked_content["epoch:train"]["init_samples"]
            samples = self.tracked_content["epoch:train"]["samples"]
            samples_info = self.tracked_content["epoch:train"]["samples_info"]
            data = self.tracked_content["epoch:train"]["data"]

            init_samples_proc = add_border(init_samples, (0.,0.5,0.), 1)
            samples_proc = []
            for cur_samples, cur_samples_info in itertools.izip(samples, samples_info):
                if cur_samples_info["new"]:
                    cur_samples_proc = add_border(cur_samples, (0.,0.5,0.), 1)     # Green border
                elif cur_samples_info["epoch_start"]:
                    cur_samples_proc = add_border(cur_samples, (1.,1.,1.), 1)     # White border
                else:
                    cur_samples_proc = cur_samples
                samples_proc.append(cur_samples_proc)

            if len(samples) < 8:
                samples_proc_padded = [np.zeros(init_samples_proc.shape).astype(np.float32)]*(8-len(samples_proc)) + samples_proc
            else:
                samples_proc_padded = samples_proc

            train_images = np.concatenate((init_samples_proc[None,:,:,:,:], np.array(samples_proc_padded), data[None,:,:,:,:]), axis=0)
            train_images = np.reshape(np.transpose(train_images, (1, 0, 2, 3, 4)), (train_images.shape[0]*train_images.shape[1], train_images.shape[2], train_images.shape[3], train_images.shape[4]))

            # 30 test images
            self._set_to_eval_mode()
            if "test_z" not in self.tracked_content["epoch:train"]:
                test_z = torch.randn(*((30,)+self.z_dim)).to(device=config.device)
                self.tracked_content["epoch:train"]["test_z"] = test_z
            else:
                test_z = self.tracked_content["epoch:train"]["test_z"]
            test_images = self.model(test_z.to(device=config.device)).cpu().data.numpy()
            self._set_to_train_mode()

            images = np.concatenate((train_images, test_images), axis=0)
            images = torch.from_numpy(images)

            tiled_image = vutils.make_grid(images, nrow=10)
            tiled_image = np.transpose(tiled_image, (1,2,0))

            if event == "prog":
                file_name = "%splot_%04d_%04d.pdf" % (config.path_prefix, context["epoch"], context["batch_idx"])
            elif event == "end":
                file_name = "%splot_%04d_end.pdf" % (config.path_prefix, context["epoch"])

            fig = plt.figure()
            plt.imshow(tiled_image, interpolation='nearest')
            fig.savefig(file_name)
            plt.close(fig)
            print("Plot saved to %s. " % (file_name))

    def train(self, dataset, hyperparams, config):

        global save_now
        global plot_now

        signal.signal(signal.SIGINT, signal_handler_menu)

        batch_size = hyperparams.batch_size
        sample_db_size = hyperparams.sample_db_size

        num_threads = config.num_threads
        device = config.device

        checkpoint_interval = config.checkpoint_interval
        plot_interval = config.plot_interval
        plot_subinterval = config.plot_subinterval
        shuffle_data = config.seed is None and config.shuffle_data

        num_samples = len(dataset) * hyperparams.num_samples_factor

        loss_fn = nn.MSELoss(reduction='sum').to(device=device)
        self._set_to_train_mode()

        seq_data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_threads-1)

        (cur_batch_data,) = next(iter(seq_data_loader))
        assert(cur_batch_data.dtype == torch.float32)

        cur_z_data_file = None

        output_dir = os.path.dirname(config.path_prefix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

		if self.testing_only:
			print("Warning: Loaded model weights, but did not load optimizer state or epoch number. Proceed anyway?")
			self._user_confirm()

        for epoch in range(self.starting_epoch, config.num_epochs):

            print("Epoch %d: %d, %d, Starting? %s Key Epoch? %s" % (epoch, self.starting_epoch, epoch % hyperparams.decay_step, repr(epoch == self.starting_epoch), repr(epoch % hyperparams.decay_step == 0)))
            print(epoch)
            print(hyperparams.decay_step)
            if epoch == self.starting_epoch or epoch % hyperparams.decay_step == 0:
                lr = hyperparams.base_lr * hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
                self.set_lr(lr)

            if epoch == 0:
                dataset_subset = ChoppedDataset(dataset, 100)
                gen_samples, _ = self.test(100, hyperparams, config)
                real_data = np.array([dataset_subset[i][0].numpy() for i in range(len(dataset_subset))])
                gen_samples_mean = np.mean(gen_samples)
                real_data_mean = np.mean(real_data)
                gen_samples_std = np.std(gen_samples)
                real_data_std = np.std(real_data)
                print("Generated Samples Mean and Std: %.4f +/- %.4f" % (gen_samples_mean,gen_samples_std))
                print("Real Data Mean and Std: %.4f +/- %.4f" % (real_data_mean,real_data_std))
                if np.abs(gen_samples_mean / real_data_mean) > 10. or np.abs(gen_samples_mean / real_data_mean) < 0.1:
                    print("Scale of the initial generated samples is not right; you should adjust the scale of initialization of the parameters. Proceed anyway?")
                    self._user_confirm()

                del gen_samples

            if epoch == self.starting_epoch or epoch % hyperparams.staleness == 0:

                if epoch % hyperparams.staleness == 0:

                    cur_z_data_file = "%sz_data_epoch_%d.npy" % (config.path_prefix, epoch)
                    if epoch == self.starting_epoch and os.path.isfile(cur_z_data_file):

                        selected_z_np = np.load(cur_z_data_file)
                        print("Loaded database of selected z's; skipping sample generation. ")

                    else:

                        self._set_to_eval_mode()

                        selected_z_np = np.empty((len(dataset),)+self.z_dim, dtype=np.float32)
                        selected_dists_np = np.tile(np.inf, (len(dataset),))

                        num_sample_dbs = int(math.ceil(num_samples / float(sample_db_size)))

                        self._track("epoch:samp", "start", {"epoch": epoch, "num_dbs": num_sample_dbs}, {}, config)

                        for j in range(num_sample_dbs):

                            if j == num_sample_dbs - 1:
                                cur_sample_db_size = (num_samples - 1) % sample_db_size + 1
                            else:
                                cur_sample_db_size = sample_db_size

                            z_np = np.empty((cur_sample_db_size,)+self.z_dim, dtype=np.float32)
                            samples_np = None
                            print("Sample DB %d out of %d: Generating samples" % (j+1, num_sample_dbs))

                            num_sample_batches = int(math.ceil(cur_sample_db_size / float(batch_size)))

                            self._track("epoch:samp:db_gen", "start", {"epoch": epoch, "db_idx": j, "cur_num_batches": num_sample_batches, "cur_num_samples": cur_sample_db_size}, {}, config)

                            for i in tqdm(range(num_sample_batches)):
                                if i == num_sample_batches - 1:
                                    cur_sample_batch_size = (cur_sample_db_size - 1) % batch_size + 1
                                else:
                                    cur_sample_batch_size = batch_size

                                cur_batch = slice(i*batch_size, i*batch_size+cur_sample_batch_size)
                                cur_batch_z = torch.randn(*((cur_sample_batch_size,)+self.z_dim)).to(device=device)
                                cur_batch_samples = self.model(cur_batch_z)
                                cur_batch_z_np = cur_batch_z.cpu().data.numpy()
                                cur_batch_samples_np = cur_batch_samples.cpu().data.numpy()
                                z_np[cur_batch] = cur_batch_z_np
                                if samples_np is None:
                                    samples_np = np.empty((cur_sample_db_size,)+cur_batch_samples.size()[1:])
                                samples_np[cur_batch] = cur_batch_samples_np

                                self._track("epoch:samp:db_gen:iter", "prog", {"epoch": epoch, "db_idx": j, "batch_idx": i, "sample_idx": cur_batch}, {"cur_batch_z": cur_batch_z_np, "cur_batch_samples": cur_batch_samples_np}, config)

                            self._track("epoch:samp:db_gen", "end", {"epoch": epoch, "db_idx": j}, {"samples": samples_np}, config)

                            samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], -1))

                            if self.dci_db is None:

                                self.dci_db = DCI(samples_flat_np.shape[1], num_comp_indices = 2, num_simp_indices = 7)

                            print("Sample DB %d out of %d: Finding nearest samples" % (j+1, num_sample_dbs))

                            self.dci_db.reset()
                            self.dci_db.add(samples_flat_np, num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002, fix_nonbase_array = True)

                            self._track("epoch:samp:db_search", "start", {"epoch": epoch, "db_idx": j, "num_batches": len(seq_data_loader), "num_data_pts": len(dataset)}, {"samples_flat": samples_flat_np, "seq_data_loader": seq_data_loader}, config)

                            for i, (cur_batch_data,) in enumerate(tqdm(seq_data_loader)):

                                cur_batch_data_flat_np = np.reshape(cur_batch_data.cpu().data.numpy().astype(np.float64), (cur_batch_data.size()[0], -1))
                                nearest_indices, nearest_dists = self.dci_db.query(cur_batch_data_flat_np, num_neighbours = 1, field_of_view = 10, prop_to_retrieve = 0.25)
                                nearest_indices = np.array(nearest_indices)[:,0]
                                nearest_dists = np.array(nearest_dists)[:,0]

                                cur_batch = slice(i*batch_size, i*batch_size+cur_batch_data.size()[0])
                                to_update = np.nonzero(nearest_dists < selected_dists_np[cur_batch])[0]

                                selected_dists_np[i*batch_size+to_update] = nearest_dists[to_update]
                                selected_z_np[i*batch_size+to_update] = z_np[nearest_indices[to_update]]

                                self._track("epoch:samp:db_search:iter", "prog", {"epoch": epoch, "db_idx": j, "batch_idx": i, "sample_idx": cur_batch}, {"cur_batch_z": lambda: z_np[nearest_indices], "cur_batch_samples": lambda: samples_np[nearest_indices], "cur_batch_data": lambda: cur_batch_data.cpu().data.numpy(), "cur_batch_dists": nearest_dists}, config)

                            self._track("epoch:samp:db_search", "end", {"epoch": epoch, "db_idx": j}, {"selected_z": selected_z_np, "selected_dists": selected_dists_np}, config)

                            #del z_np, samples_np, samples_flat_np

                        selected_z_np += 0.1*np.random.randn(*selected_z_np.shape)

                        self._track("epoch:samp", "end", {"epoch": epoch}, {"selected_z": selected_z_np, "selected_dists": selected_dists_np}, config)

                        del selected_dists_np

                        #cur_z_data_file = "%sz_data_epoch_%d.npy" % (config.path_prefix, epoch)
                        np.save(cur_z_data_file, selected_z_np)

                    self._reset_optimizer()

                else:

                    assert(self.init_z_data is not None)
                    selected_z_np = self.init_z_data
                    cur_z_data_file = self.init_z_data_file
                    self.init_z_data = None
                    self.init_z_data_file = None

                comb_dataset = ZippedDataset(dataset, data.TensorDataset(torch.from_numpy(selected_z_np)))

                data_loader = data.DataLoader(comb_dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=num_threads-1, pin_memory=True)

                self._set_to_train_mode()


            err = 0.
            losses = deque(maxlen=100)
            progress = tqdm(total=len(data_loader), desc='Epoch % 3d' % epoch)

            self._track("epoch:train", "start", {"epoch": epoch, "num_batches": len(data_loader), "num_data_pts": len(comb_dataset)}, {"comb_dataset": comb_dataset, "data_loader": data_loader}, config)

            for i, ((cur_batch_data,), (cur_batch_z,)) in enumerate(data_loader):

                cur_batch_data = cur_batch_data.to(device=device, non_blocking=True)
                cur_batch_z = cur_batch_z.to(device=device)

                self.model.zero_grad()
                cur_batch_samples = self.model(cur_batch_z)
                loss = loss_fn(cur_batch_samples, cur_batch_data) / float(2*cur_batch_data.size(0))
                loss.backward()

                err += loss.item()*cur_batch_data.size()[0]
                self.optimizer.step()

                losses.append(loss.item())
                recent_loss = np.mean(losses)
                progress.set_postfix({'Recent loss': recent_loss})
                progress.update()

                self._track("epoch:train:iter", "prog", {"epoch": epoch, "batch_idx": i}, {"cur_batch_z": lambda: cur_batch_z.cpu().data.numpy(), "cur_batch_samples": lambda: cur_batch_samples.cpu().data.numpy(), "cur_batch_data": lambda: cur_batch_data.cpu().data.numpy(), "cur_batch_size": cur_batch_data.size()[0], "loss": losses[-1], "recent_loss": recent_loss}, config)

                if (plot_subinterval is not None and (i + 1) % plot_subinterval == 0) or plot_now:
                    plot_now = False
                    self.plot("epoch:train:iter", "prog", {"epoch": epoch, "batch_idx": i}, config)
                    signal.signal(signal.SIGINT, signal_handler_menu)

            progress.close()

            self._track("epoch:train", "end", {"epoch": epoch}, {}, config)

            err /= len(dataset)
            print("Epoch %d: Error over entire dataset: %f" % (epoch, err))

            if (checkpoint_interval is not None and (epoch + 1) % checkpoint_interval == 0) or epoch == config.num_epochs - 1 or save_now:
                save_now = False
                self.save(epoch, hyperparams, config, {'lr': lr, 'err': err, 'z_data_file': cur_z_data_file})

            if (plot_interval is not None and (epoch + 1) % plot_interval == 0) or epoch == config.num_epochs - 1 or plot_now:
                plot_now = False
                self.plot("epoch:train", "end", {"epoch": epoch}, config)
                signal.signal(signal.SIGINT, signal_handler_menu)

        signal.signal(signal.SIGINT, signal_handler_noop)


    # Difference from sample(): this can generate samples in batches and is suitable for generating a large number of samples
    def test(self, num_samples, hyperparams, config, keep_z = False):
        batch_size = hyperparams.batch_size
        device = config.device

        self._set_to_eval_mode()

        samples_np = None
        if keep_z:
            z_np = np.empty((num_samples,)+self.z_dim, dtype=np.float32)
        else:
            z_np = None

        num_sample_batches = int(math.ceil(num_samples / float(batch_size)))
        for i in tqdm(range(num_sample_batches)):
            if i == num_sample_batches - 1:
                cur_sample_batch_size = (num_samples - 1) % batch_size + 1
            else:
                cur_sample_batch_size = batch_size

            z = torch.randn(*((cur_sample_batch_size,)+self.z_dim)).to(device=device)
            samples = self.model(z)
            if keep_z:
                z_np[i*batch_size:i*batch_size+cur_sample_batch_size] = z.cpu().data.numpy()
            if samples_np is None:
                samples_np = np.empty((num_samples,)+samples.size()[1:], dtype=np.float32)
            samples_np[i*batch_size:i*batch_size+cur_sample_batch_size] = samples.cpu().data.numpy()

        return samples_np, z_np

    def save(self, epoch, hyperparams, config, aux_info):

        file_name = "%scheckpoint_epoch_%d.pth.tar" % (config.path_prefix, epoch)
        data = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'hyperparams': hyperparams, 'config': config, 'checkpoint_file': file_name}
        data.update(aux_info)

        torch.save(data, file_name)
        print("Checkpoint saved to %s. " % (file_name))

    def restore(self, file_path, testing_only = False):

        if os.path.isfile(file_path):
            print("Loading checkpoint from %s" % (file_path))
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['state_dict'])

            if not testing_only:

            	self.starting_epoch = checkpoint['epoch'] + 1
            	print("Next epoch is %d" % (self.starting_epoch))

				self.optimizer.load_state_dict(checkpoint['optimizer'])

				self.testing_only = False

				if os.path.basename(checkpoint['checkpoint_file']) == os.path.basename(checkpoint['z_data_file']):
					# Automatically adjust z-data file path if the checkpoint file was moved
					z_data_file_name = os.path.join(os.path.dirname(file_path), os.path.basename(checkpoint['z_data_file']))
				else:
					# Don't adjust z-data file path if the checkpoint file was originally using a z-data file from another directory (which may happen if the checkpoint file was moved to somewhere else, and training was resumed from that point and terminated again before new z-data was generated)
					z_data_file_name = checkpoint['z_data_file']
				if os.path.isfile(z_data_file_name):
					self.init_z_data = np.load(z_data_file_name)
					self.init_z_data_file = z_data_file_name
				else:
					print("Warning: File containing z used in the previous epoch cannot be found at \"%s\". Previous z's not loaded. " % (z_data_file_name))
					self.init_z_data = None
					self.init_z_data_file = None
			else:

				self.testing_only = True

        else:
            print("No checkpoint found at %s" % (file_path))

    def forward(self, z_vec, device):
        self._set_to_eval_mode()
        s = self.model(torch.from_numpy(z_vec).float().to(device=device))#.data.cpu().numpy()
        return s


class CondIMLE():
    # z_dim is a tuple, e.g.: (C, H, W)
    def __init__(self, z_dim, model, hyperparams, config, freeze_batch_norm = True):
        self.z_dim = z_dim
        self.model = model.to(device=config.device)
        if hasattr(self.model, "get_initializer"):
            self.model.apply(self.model.get_initializer())

        if hyperparams.optimizer == "adam":
        	self.optimizer = optim.Adam(self.model.parameters(), lr=hyperparams.base_lr, betas=hyperparams.momentum, weight_decay=0.0005)
        elif hyperparams.optimizer == "sgd":
        	self.optimizer = optim.SGD(self.model.parameters(), lr=hyperparams.base_lr, momentum=hyperparams.momentum, weight_decay=0.0005)
        else:
        	raise Exception("Unknown optimizer: optimizer must be either 'adam' or 'sgd'.")

        if not config.disable_cuda and torch.cuda.is_available():
            if not config.disable_multigpu and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        self.dci_db = None
        self.starting_epoch = 0
        self.init_z_data = None
        self.init_z_data_file = None
        self.testing_only = False		# If model weights are loaded but optimizer state is not loaded
        self.freeze_batch_norm = freeze_batch_norm
        self.tracked_context = dict()   # Populated automatically
        self.tracked_content = dict()   # Populated manually


    def set_lr(self, lr):
        # Set new learning rate
        for i, group in enumerate(self.optimizer.param_groups):
        	print("Original learning rate for parameter group %d is %f; changed to %f and zeroed gradient statistics" % (i, group['lr'], lr))
            if lr != group['lr']:
                #print("Original learning rate for parameter group %d is %f; changed to %f and zeroed gradient statistics" % (i, group['lr'], lr))
                group['lr'] = lr
                for p in group['params']:
                    # Reset optimizer state
                    del self.optimizer.state[p]
                    self.optimizer.state[p] = dict()

    def _set_to_train_mode(self):
        self.model.train()

        if self.freeze_batch_norm:

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    #print("Setting %s layer to eval mode" % (classname))
                    m.eval()

            self.model.apply(set_bn_eval)

    def _set_to_eval_mode(self):
        self.model.eval()

    def _reset_optimizer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # Reset optimizer state
                if p in self.optimizer.state:
                    del self.optimizer.state[p]
                    self.optimizer.state[p] = dict()

    def _user_confirm(self):
        while True:
            option = raw_input("(Y)es or (N)o? ")
            if option.lower() == "y":
                break
            elif option.lower() == "n":
                raise KeyboardInterrupt
            else:
                print("Invalid option.")

    def _track(self, loc, event, cur_context, cur_content, config):
        self.tracked_context[loc] = cur_context
        context = dict()
        dirs = loc.split(":")
        for i in range(len(dirs)):
            cur_key = ":".join(dirs[:i+1])
            if cur_key in self.tracked_context:
                context.update(self.tracked_context[cur_key])

        content = dict()
        for i in range(len(dirs)):
            cur_key = ":".join(dirs[:i+1])
            if cur_key in self.tracked_content:
                content.update(self.tracked_content[cur_key])
        content.update(cur_content)

        lambda_content = dict()
        for key in content:
            if callable(content[key]):
                lambda_content[key] = content[key]
            else:
                lambda_content[key] = lambda key=key: content[key]

        self.track(loc, event, context, lambda_content, config)

        if event == "end":
            del self.tracked_context[loc]

    def track(self, loc, event, context, content, config):
        if loc == "epoch:samp" and event == "end":
            if loc not in self.tracked_content:
                self.tracked_content[loc] = dict()
            self.tracked_content[loc]["is_new"] = True
        elif loc.startswith("epoch:train"):
            if loc == "epoch:train" and event == "start":
                comb_dataset = content["comb_dataset"]()
                data_loader = data.DataLoader(comb_dataset, batch_size=7, shuffle=False, num_workers=1, pin_memory=False)
                (cur_batch_data,), (cur_batch_z,) = next(iter(data_loader))
                if loc not in self.tracked_content:
                    self.tracked_content[loc] = dict()
                self.tracked_content[loc]["z"]  = np.copy(cur_batch_z.cpu().data.numpy())
                self.tracked_content[loc]["data"] = np.copy(cur_batch_data.cpu().data.numpy())
                if "samples" not in self.tracked_content[loc]:
                    self.tracked_content[loc]["samples"] = deque(maxlen=8)
                    self.tracked_content[loc]["samples_info"] = deque(maxlen=8)
                self._set_to_eval_mode()
                cur_batch_samples = self.model(cur_batch_z.to(device=config.device)).cpu().data.numpy()
                self._set_to_train_mode()
                is_new = "epoch:samp" not in self.tracked_content or self.tracked_content["epoch:samp"]["is_new"]
                if is_new:
                    if "epoch:samp" not in self.tracked_content:
                         self.tracked_content["epoch:samp"] = dict()
                    self.tracked_content["epoch:samp"]["is_new"] = False
                    self.tracked_content[loc]["init_samples"] = cur_batch_samples
                self.tracked_content[loc]["samples"].append(cur_batch_samples)
                self.tracked_content[loc]["samples_info"].append({"epoch_start": True, "new": is_new})
            elif loc == "epoch:train:iter" and event == "prog":
                cur_batch_z = torch.from_numpy(content["z"]())
                self._set_to_eval_mode()
                cur_batch_samples = self.model(cur_batch_z.to(device=config.device)).cpu().data.numpy()
                self._set_to_train_mode()
                batch_idx = context["batch_idx"]
                if (batch_idx+1) % config.track_subinterval == 0:
                    self.tracked_content["epoch:train"]["samples"].append(cur_batch_samples)
                    self.tracked_content["epoch:train"]["samples_info"].append({"epoch_start": False, "new": False})
            elif loc == "epoch:train" and event == "end":
                del self.tracked_context["%s:iter" % (loc)]

    def plot(self, loc, event, context, config):
        def add_border(imgs, colour, border_width):
            imgs_proc = np.copy(imgs)
            assert(len(colour) == 3)
            for i in range(len(colour)):
                imgs_proc[:,i,:,:border_width] = colour[i]  # Left border
                imgs_proc[:,i,:,-border_width:] = colour[i] # Right border
                imgs_proc[:,i,:border_width,:] = colour[i]  # Top border
                imgs_proc[:,i,-border_width:,:] = colour[i] # Bottom border
            return imgs_proc

        if loc.startswith("epoch:train") and "epoch:train" in self.tracked_content:

            init_samples = self.tracked_content["epoch:train"]["init_samples"]
            samples = self.tracked_content["epoch:train"]["samples"]
            samples_info = self.tracked_content["epoch:train"]["samples_info"]
            data = self.tracked_content["epoch:train"]["data"]

            init_samples_proc = add_border(init_samples, (0.,0.5,0.), 1)
            samples_proc = []
            for cur_samples, cur_samples_info in itertools.izip(samples, samples_info):
                if cur_samples_info["new"]:
                    cur_samples_proc = add_border(cur_samples, (0.,0.5,0.), 1)     # Green border
                elif cur_samples_info["epoch_start"]:
                    cur_samples_proc = add_border(cur_samples, (1.,1.,1.), 1)     # White border
                else:
                    cur_samples_proc = cur_samples
                samples_proc.append(cur_samples_proc)

            if len(samples) < 8:
                samples_proc_padded = [np.zeros(init_samples_proc.shape).astype(np.float32)]*(8-len(samples_proc)) + samples_proc
            else:
                samples_proc_padded = samples_proc

            train_images = np.concatenate((init_samples_proc[None,:,:,:,:], np.array(samples_proc_padded), data[None,:,:,:,:]), axis=0)
            train_images = np.reshape(np.transpose(train_images, (1, 0, 2, 3, 4)), (train_images.shape[0]*train_images.shape[1], train_images.shape[2], train_images.shape[3], train_images.shape[4]))

            # 30 test images
            self._set_to_eval_mode()
            if "test_z" not in self.tracked_content["epoch:train"]:
                test_z = torch.randn(*((30,)+self.z_dim)).to(device=config.device)
                self.tracked_content["epoch:train"]["test_z"] = test_z
            else:
                test_z = self.tracked_content["epoch:train"]["test_z"]
            test_images = self.model(test_z.to(device=config.device)).cpu().data.numpy()
            self._set_to_train_mode()

            images = np.concatenate((train_images, test_images), axis=0)
            images = torch.from_numpy(images)

            tiled_image = vutils.make_grid(images, nrow=10)
            tiled_image = np.transpose(tiled_image, (1,2,0))

            if event == "prog":
                file_name = "%splot_%04d_%04d.pdf" % (config.path_prefix, context["epoch"], context["batch_idx"])
            elif event == "end":
                file_name = "%splot_%04d_end.pdf" % (config.path_prefix, context["epoch"])

            fig = plt.figure()
            plt.imshow(tiled_image, interpolation='nearest')
            fig.savefig(file_name)
            plt.close(fig)
            print("Plot saved to %s. " % (file_name))

    # dataset[i] should return a tuple (input, output)
    def train(self, dataset, hyperparams, config):

		global save_now
        global plot_now

        signal.signal(signal.SIGINT, signal_handler_menu)

        batch_size = hyperparams.batch_size
        num_samples_factor = hyperparams.num_samples_factor

        num_threads = config.num_threads
        device = config.device

        checkpoint_interval = config.checkpoint_interval
        plot_interval = config.plot_interval
        plot_subinterval = config.plot_subinterval
        shuffle_data = config.seed is None and config.shuffle_data

        loss_fn = nn.MSELoss().to(device=device)
        self._set_to_train_mode()

        seq_data_loader = data.DataLoader(dataset, batch_size=1, num_workers=num_threads-1, pin_memory=True)

        cur_batch_input, cur_batch_output = next(iter(seq_data_loader))
        assert(cur_batch_input.dtype == torch.float32)
        assert(cur_batch_output.dtype == torch.float32)

        cur_z_data_file = None

        output_dir = os.path.dirname(config.path_prefix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

		if self.testing_only:
			print("Warning: Loaded model weights, but did not load optimizer state or epoch number. Proceed anyway?")
			self._user_confirm()

        for epoch in range(self.starting_epoch, config.num_epochs):

            if epoch == self.starting_epoch or epoch % hyperparams.decay_step == 0:
                lr = hyperparams.base_lr * hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
                self.set_lr(lr)

            if epoch == 0:
                dataset_subset = ChoppedDataset(dataset, 100)
                dataset_subset_input = SlicedDataset(dataset_subset, (0,))
                dataset_subset_output = SlicedDataset(dataset_subset, (1,))
                gen_samples, _ = self.test(dataset_subset, 1, hyperparams, config)
                assert(gen_samples.shape[1] == 1)
                gen_samples = gen_samples[:,0]
                real_data = np.array([dataset_subset_output[i][0].numpy() for i in range(len(dataset_subset_output))])
                gen_samples_mean = np.mean(gen_samples)
                real_data_mean = np.mean(real_data)
                gen_samples_std = np.std(gen_samples)
                real_data_std = np.std(real_data)
                print("Generated Samples Mean and Std: %.4f +/- %.4f" % (gen_samples_mean,gen_samples_std))
                print("Real Data Mean and Std: %.4f +/- %.4f" % (real_data_mean,real_data_std))
                if np.abs(gen_samples_mean / real_data_mean) > 10. or np.abs(gen_samples_mean / real_data_mean) < 0.1:
                    print("Scale of the initial generated samples is not right; you should adjust the scale of initialization of the parameters. Proceed anyway?")
                    self._user_confirm()

                del gen_samples

            if epoch == self.starting_epoch or epoch % hyperparams.staleness == 0:

                if epoch % hyperparams.staleness == 0:

                    self._set_to_eval_mode()
                    selected_z_np = np.empty((len(dataset),)+self.z_dim, dtype=np.float32)

                    print("Generating samples and finding nearest samples")

                    for i, (cur_batch_input, cur_batch_output) in enumerate(tqdm(seq_data_loader)):

                        z = torch.randn(*((num_samples_factor,)+self.z_dim)).to(device=device)
                        cur_batch_input = cur_batch_input.to(device=device, non_blocking=True).expand(num_samples_factor, -1, -1, -1)
                        samples = self.model(cur_batch_input, z)

                        samples_np = samples.cpu().data.numpy().astype(np.float64)
                        samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], -1))

                        if self.dci_db is None:
                            self.dci_db = DCI(samples_flat_np.shape[1], num_comp_indices = 1, num_simp_indices = 3)

                        self.dci_db.reset()
                        self.dci_db.add(samples_flat_np, num_levels = 1, field_of_view = 10, prop_to_retrieve = 0.002, fix_nonbase_array = True)

                        cur_batch_output_flat_np = np.reshape(cur_batch_output.cpu().data.numpy().astype(np.float64), (cur_batch_output.size()[0], -1))
                        nearest_indices, _ = self.dci_db.query(cur_batch_output_flat_np, num_neighbours = 1, field_of_view = 10, prop_to_retrieve = 0.1)
                        nearest_index = nearest_indices[0][0]

                        selected_z_np[i] = z[nearest_index].cpu().data.numpy()

                        del samples_np, samples_flat_np

                    cur_z_data_file = "%sz_data_epoch_%d.npy" % (config.path_prefix, epoch)
                    np.save(cur_z_data_file, selected_z_np)

                    self._reset_optimizer()

                else:

                    assert(self.init_z_data is not None)
                    selected_z_np = self.init_z_data
                    cur_z_data_file = self.init_z_data_file
                    self.init_z_data = None
                    self.init_z_data_file = None

                comb_dataset = ZippedDataset(dataset, data.TensorDataset(torch.from_numpy(selected_z_np)))

                data_loader = data.DataLoader(comb_dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=num_threads-1, pin_memory=True)

                self._set_to_train_mode()

            err = 0.
            losses = deque(maxlen=100)
            progress = tqdm(total=len(data_loader), desc='Epoch % 3d' % epoch)

            for i, ((cur_batch_input, cur_batch_output), (cur_batch_z,)) in enumerate(data_loader):

                cur_batch_input = cur_batch_input.to(device=device, non_blocking=True)
                cur_batch_output = cur_batch_output.to(device=device, non_blocking=True)
                cur_batch_z = cur_batch_z.to(device=device)

                self.model.zero_grad()
                cur_batch_samples = self.model(cur_batch_input, cur_batch_z)
                loss = loss_fn(cur_batch_samples, cur_batch_output)
                loss.backward()
                err += loss.item()*cur_batch_output.size()[0]
                self.optimizer.step()

                losses.append(loss.item())
                progress.set_postfix({'Recent loss': np.mean(losses)})
                progress.update()

            progress.close()

            err /= len(dataset)
            print("Epoch %d: Error: %f" % (epoch, err))

            if (checkpoint_interval is not None and (epoch + 1) % checkpoint_interval == 0) or epoch == config.num_epochs - 1 or save_now:
                self.save(epoch, hyperparams, config, {'lr': lr, 'err': err, 'z_data_file': cur_z_data_file})

    # Difference from sample(): this can generate samples in batches and is suitable for generating a large number of samples
    # num_samples_factor is the number of samples generated per input example from the dataset
    def test(self, dataset, num_samples_factor, hyperparams, config, keep_z = False):
        batch_size = hyperparams.batch_size
        device = config.device
        num_threads = config.num_threads

        num_inputs_per_batch = batch_size // num_samples_factor
        seq_data_loader = data.DataLoader(dataset, batch_size=num_inputs_per_batch, num_workers=num_threads-1, pin_memory=True)

        self._set_to_eval_mode()

        samples_np = None
        if keep_z:
            z_np = np.empty((len(dataset), num_samples_factor)+self.z_dim, dtype=np.float32)
        else:
            z_np = None

        for i, (cur_batch_input,) in enumerate(tqdm(seq_data_loader)):

            z = torch.randn(*((cur_batch_input.size(0)*num_samples_factor,)+self.z_dim)).to(device=device)
            cur_batch_input = cur_batch_input.to(device=device, non_blocking=True).unsqueeze(1).expand(-1, num_samples_factor, -1, -1, -1).view(*((cur_batch_input.size(0)*num_samples_factor,)+cur_batch_input.size()[1:]))
            samples = self.model(cur_batch_input, z)

            if keep_z:
                z_np[i*num_inputs_per_batch:i*num_inputs_per_batch+cur_batch_input.size(0)] = z.view(*((cur_batch_input.size(0),num_samples_factor)+self.z_dim)).cpu().data.numpy()
            if samples_np is None:
                samples_np = np.empty((len(dataset),num_samples_factor)+samples.size()[1:], dtype=np.float32)
            samples_np[i*num_inputs_per_batch:i*num_inputs_per_batch+cur_batch_input.size(0)] = samples.view(*((cur_batch_input.size(0), num_samples_factor)+samples.size()[1:])).cpu().data.numpy()

        return samples_np, z_np

    def save(self, epoch, hyperparams, config, aux_info):

        file_name = "%scheckpoint_epoch_%d.pth.tar" % (config.path_prefix, epoch)
        data = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'hyperparams': hyperparams, 'config': config, 'checkpoint_file': file_name}
        data.update(aux_info)

        torch.save(data, file_name)
        print("Checkpoint saved to %s. " % (file_name))

    def restore(self, file_path, testing_only = False):

        if os.path.isfile(file_path):
            print("Loading checkpoint from %s" % (file_path))
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['state_dict'])

            if not testing_only:

            	self.starting_epoch = checkpoint['epoch'] + 1
            	print("Next epoch is %d" % (self.starting_epoch))

				self.optimizer.load_state_dict(checkpoint['optimizer'])

				self.testing_only = False

				if os.path.basename(checkpoint['checkpoint_file']) == os.path.basename(checkpoint['z_data_file']):
					# Automatically adjust z-data file path if the checkpoint file was moved
					z_data_file_name = os.path.join(os.path.dirname(file_path), os.path.basename(checkpoint['z_data_file']))
				else:
					# Don't adjust z-data file path if the checkpoint file was originally using a z-data file from another directory (which may happen if the checkpoint file was moved to somewhere else, and training was resumed from that point and terminated again before new z-data was generated)
					z_data_file_name = checkpoint['z_data_file']
				if os.path.isfile(z_data_file_name):
					self.init_z_data = np.load(z_data_file_name)
					self.init_z_data_file = z_data_file_name
				else:
					print("Warning: File containing z used in the previous epoch cannot be found at \"%s\". Previous z's not loaded. " % (z_data_file_name))
					self.init_z_data = None
					self.init_z_data_file = None
			else:

				self.testing_only = True

        else:
            print("No checkpoint found at %s" % (file_path))

    def forward(self, input_vec, z_vec, device):
        self._set_to_eval_mode()
        s = self.model(torch.from_numpy(input_vec).float().to(device=device), torch.from_numpy(z_vec).float().to(device=device))#cpu().data.numpy()
        return s

def main():

    parser = argparse.ArgumentParser(description='IMLE Trainer')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--disable-multigpu', action='store_true', help='Disable data parallelism')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--threads', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('-n', '--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-o', '--out-prefix', default='checkpoints/', type=str, metavar='STR',
                    help='prefix for checkpoint path')
    parser.add_argument('-f', '--check-freq', default=5, type=int, metavar='N',
                    help='the number of epochs that pass before a checkpoint is saved')

    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device('cuda:%d' % (args.gpu))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Hyperparameters:

    # base_lr: Base learning rate
    # batch_size: Batch size
    # sample_db_size: Maximum number of samples that are kept in memory for nearest neighbour search
    # decay_step: Number of epochs before learning rate decay
    # decay_rate: Rate of learning rate decay
    # staleness: Number of times to re-use nearest samples
    # num_samples_factor: Ratio of the number of generated samples to the number of real data examples
    hyperparams = Hyperparams(base_lr=1e-3, batch_size=64, sample_db_size=1024, decay_step=25, decay_rate=1.0, staleness=10, num_samples_factor=10)
    #hyperparams = Hyperparams(base_lr=1e-3, batch_size=64, sample_db_size=1024, decay_step=1, decay_rate=0.5, staleness=10, num_samples_factor=10)

    config = Config(num_epochs=args.epochs, shuffle_data=True, path_prefix=args.out_prefix, checkpoint_interval=args.check_freq, device=args.device, disable_cuda=args.disable_cuda, disable_multigpu=args.disable_multigpu, num_threads=args.threads, seed=args.seed)

    # train_input and train_output are of shape N x C x H x W, where N is the number of examples, C is the number of channels, H is the height and W is the width
    train_input = np.random.randn(128, 10, 1, 1).astype(np.float32)
    train_output = np.random.randn(128, 1, 28, 28).astype(np.float32)

    input_dim = (10,1,1)
    z_dim = (64,1,1)

    imle = IMLE(z_dim, ConvImplicitModel(z_dim), hyperparams, config)
    #imle = CondIMLE(z_dim, CondConvImplicitModel(input_dim, z_dim), hyperparams, config)

    if args.resume:
        imle.restore(args.resume)

    imle.train(data.TensorDataset(torch.from_numpy(train_output)), hyperparams, config)
    #imle.train(data.TensorDataset(torch.from_numpy(train_input), torch.from_numpy(train_output)), hyperparams, config)


if __name__ == '__main__':
    main()
