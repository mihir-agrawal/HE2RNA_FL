"""
HE2RNA: definition of the algorithm to generate a model for gene expression prediction
Copyright (C) 2020  Owkin Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import torch
import time
import os
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm


class HE2RNA(nn.Module):
    """Model that generates one score per tile and per predicted gene.

    Args
        output_dim (int): Output dimension, must match the number of genes to
            predict.
        layers (list): List of the layers' dimensions
        nonlin (torch.nn.modules.activation)
        ks (list): list of numbers of highest-scored tiles to keep in each
            channel.
        dropout (float)
        device (str): 'cpu' or 'cuda'
        mode (str): 'binary' or 'regression'
    """
    def __init__(self, input_dim, output_dim,
                 layers=[1], nonlin=nn.ReLU(), ks=[10],
                 dropout=0.5, device='cpu',
                 bias_init=None, **kwargs):
        super(HE2RNA, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = [input_dim] + layers + [output_dim]
        self.layers = []
        for i in range(len(layers) - 1):
            layer = nn.Conv1d(in_channels=layers[i],
                              out_channels=layers[i+1],
                              kernel_size=1,
                              stride=1,
                              bias=True)
            setattr(self, 'conv' + str(i), layer)
            self.layers.append(layer)
        if bias_init is not None:
            self.layers[-1].bias = bias_init
        self.ks = np.array(ks)

        self.nonlin = nonlin
        self.do = nn.Dropout(dropout)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        if self.training:
            k = int(np.random.choice(self.ks))
            return self.forward_fixed_k(x, k)
        else:
            pred = 0
            for k in self.ks:
                pred += self.forward_fixed_k(x, int(k)) / len(self.ks)
            return pred
    
    def forward_fixed_k(self, x, k):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("🚨 NaN/Inf in input to forward_fixed_k")

        mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = (mask > 0).float()

        x = self.conv(x)
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            # print(x)
            print("🚨 NaN/Inf after conv(x)")

    
        # move to CPU so we don’t get the MPS fallback warning
        x_cpu = x

        # build masks
        nan_mask = torch.isnan(x_cpu)
        inf_mask = torch.isinf(x_cpu)
        bad_mask = nan_mask | inf_mask

        if bad_mask.any():
            # find the (tile, channel, position) triples where x is bad
            idx = torch.nonzero(bad_mask, as_tuple=False)
            print(f"🚨 Found {len(idx)} NaN/Inf entries in x after conv:")
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("🚨 NaN/Inf after conv(x)")
            
        #     nan_mask = torch.isnan(x)
        #     inf_mask = torch.isinf(x)

        #     if nan_mask.any():
        #         print("NaN found at positions:")
        #         print(torch.nonzero(nan_mask, as_tuple=False))

        #     if inf_mask.any():
        #         print("Inf found at positions:")
        #         print(torch.nonzero(inf_mask, as_tuple=False))

        #     # Optional: Print the actual values around the issue
        #     indices = torch.nonzero(nan_mask | inf_mask, as_tuple=False)
        #     for idx in indices:
        #         print(f"x{tuple(idx.tolist())} = {x[tuple(idx.tolist())].item()}")
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("🚨 NaN/Inf after conv(x)")
            
        #     nan_mask = torch.isnan(x)
        #     inf_mask = torch.isinf(x)
        #     error_mask = nan_mask | inf_mask

        #     indices = torch.nonzero(error_mask, as_tuple=False).cpu()

        #     for idx in indices:
        #         idx_tuple = tuple(idx.tolist())
        #         value = x[idx_tuple].item()
        #         print(f"x{idx_tuple} = {value}")
        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("🚨 NaN/Inf after conv(x)")
            
        #     # Move x to CPU first to avoid MPS fallback warnings
        #     x_cpu = x.detach().to("cpu")

        #     nan_mask = torch.isnan(x_cpu)
        #     inf_mask = torch.isinf(x_cpu)
        #     error_mask = nan_mask | inf_mask

        #     if error_mask.any():
        #         indices = torch.nonzero(error_mask, as_tuple=False)
        #         print(f"Found {len(indices)} NaN/Inf values at:")
        #         for idx in indices:
        #             idx_tuple = tuple(idx.tolist())
        #             value = x_cpu[idx_tuple].item()
        #             print(f"x{idx_tuple} = {value}")
        #     else:
        #         print("Warning: NaN/Inf detected by .any(), but no indices found.")

        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     print("🚨 NaN/Inf after conv(x)")

        #     nan_mask = torch.isnan(x)
        #     inf_mask = torch.isinf(x)
        #     error_mask = nan_mask | inf_mask

        #     indices = torch.nonzero(error_mask, as_tuple=False)

        #     if indices.numel() == 0:
        #         print("Warning: NaN/Inf detected by .any(), but no nonzero() indices returned.")
        #     else:
        #         print(f"Found {len(indices)} NaN/Inf values at:")
        #         for idx in indices:
        #             idx_tuple = tuple(idx.tolist())
        #             try:
        #                 value = x[idx_tuple].item()
        #                 print(f"x{idx_tuple} = {value}")
        #             except RuntimeError:
        #                 print(f"x{idx_tuple} = Unable to read value (MPS limitation?)")

        # x_cpu = x.cpu()
        # if torch.isnan(x_cpu).any() or torch.isinf(x_cpu).any():
        #     print("🚨 NaN/Inf after conv(x)")
        #     nan_mask = torch.isnan(x_cpu)
        #     inf_mask = torch.isinf(x_cpu)
        #     error_mask = nan_mask | inf_mask
        #     indices = torch.nonzero(error_mask, as_tuple=False)
        #     if indices.numel() == 0:
        #         print("Warning: NaN/Inf detected by .any(), but no nonzero() indices returned on CPU.")
        #     else:
        #         print(f"Found {len(indices)} NaN/Inf values at:")
        #         for idx in indices:
        #             idx_tuple = tuple(idx.tolist())
        #             try:
        #                 value = x_cpu[idx_tuple].item()
        #                 print(f"x{idx_tuple} = {value}")
        #             except RuntimeError:
        #                 print(f"x{idx_tuple} = Unable to read value")





        x = x * mask

        t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)

        if torch.isnan(t).any() or torch.isinf(t).any():
            print("🚨 NaN/Inf in top-k values")

        numerator = torch.sum(t * mask[:, :, :k], dim=2)
        denominator = torch.sum(mask[:, :, :k], dim=2)

        if torch.any(denominator == 0):
            print("🚨 Zero found in denominator")

        out = numerator / denominator

        if torch.isnan(out).any() or torch.isinf(out).any():
            print("🚨 NaN or Inf in final output of forward_fixed_k")

        return out

    # def forward_fixed_k(self, x, k):
    #     with torch.autograd.detect_anomaly():

    #         # Check input tensor
    #         # if torch.isnan(x).any() or torch.isinf(x).any():
    #         #     print("🚨 NaN/Inf detected in input tensor of forward_fixed_k")
            
    #         # Compute mask from the input tensor
    #         mask, _ = torch.max(x, dim=1, keepdim=True)
    #         mask = (mask > 0).float()
    #         # if torch.isnan(mask).any() or torch.isinf(mask).any():
    #         #     print("🚨 NaN/Inf detected in computed mask")
            
    #         # Pass x through the conv layers
    #         conv_out = self.conv(x)
    #         # if torch.isnan(conv_out).any() or torch.isinf(conv_out).any():
    #         #     print("🚨 NaN/Inf detected after self.conv(x)")
            
    #         # Multiply conv output by mask
    #         x = conv_out * mask
    #         # if torch.isnan(x).any() or torch.isinf(x).any():
    #         #     print("🚨 NaN/Inf detected after multiplying conv output with mask")
            
    #         # Get top-k values along the spatial dimension
    #         t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)
    #         # if torch.isnan(t).any() or torch.isinf(t).any():
    #         #     print("🚨 NaN/Inf detected in topk values")
            
    #         # Sum over the first k elements of mask to get the denominator
    #         denominator = torch.sum(mask[:, :, :k], dim=2)
    #         # if torch.any(denominator == 0):
    #         #     print("🚨 Warning: Zero found in denominator during forward pass.")

    #         # To avoid division by zero, add a small epsilon value
    #         eps = 1e-8
    #         out = torch.sum(t * mask[:, :, :k], dim=2) / (denominator + eps)
    #         # if torch.isnan(out).any() or torch.isinf(out).any():
    #         #     print("🚨 NaN or Inf detected in final output of forward_fixed_k")
        
    #     return out

    # def forward_fixed_k(self, x, k):
    #     # Compute mask and apply conv layers
    #     mask, _ = torch.max(x, dim=1, keepdim=True)
    #     mask = (mask > 0).float()
    #     conv_out = self.conv(x)
    #     x = conv_out * mask

    #     # Debug print: check shape and values before top-k
    #     print("x shape:", x.shape)
    #     if torch.isnan(x).any():
    #         nan_indices = torch.nonzero(torch.isnan(x))
    #         print("NaNs found in x at indices:", nan_indices)
    #     if torch.isinf(x).any():
    #         inf_indices = torch.nonzero(torch.isinf(x))
    #         print("Infs found in x at indices:", inf_indices)
    #     print("x stats -- min:", x.min().item(), "max:", x.max().item(), "mean:", x.mean().item())
        
    #     # Proceed with top-k using x
    #     t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)
    #     eps = 1e-8
    #     out = torch.sum(t * mask[:, :, :k], dim=2) / (torch.sum(mask[:, :, :k], dim=2) + eps)
    #     return out



    # def forward_fixed_k(self, x, k):
    #     mask, _ = torch.max(x, dim=1, keepdim=True)
    #     mask = (mask > 0).float()
    #     x = self.conv(x) * mask
    #     t, _ = torch.topk(x, k, dim=2, largest=True, sorted=True)
    #     denominator = torch.sum(mask[:, :, :k], dim=2) 
    #     if torch.any(denominator == 0):
    #         print("🚨 Warning: Zero found in denominator during forward pass. This will lead to NaN/Inf.")

    #     x = torch.sum(t * mask[:, :, :k], dim=2) / torch.sum(mask[:, :, :k], dim=2)
    #     if torch.isnan(x).any() or torch.isinf(x).any():
    #         print("🚨 NaN or Inf detected in model forward pass")

    #     return x

    # def conv(self, x):
    #     x = x[:, x.shape[1] - self.input_dim:]
    #     for i in range(len(self.layers) - 1):
    #         x = self.do(self.nonlin(self.layers[i](x)))
    #     x = self.layers[-1](x)
    #     return x
    def conv(self, x):
        x = x[:, x.shape[1] - self.input_dim:]
        for i in range(len(self.layers) - 1):
            x = self.do(self.nonlin(self.layers[i](x)))
            # Clamp the activations to prevent exploding values:
            x = torch.clamp(x, min=-1e4, max=1e4)
        x = self.layers[-1](x)
        return x

    
    # def conv(self, x):
    # # Select only the last input_dim channels
    #     x = x[:, x.shape[1] - self.input_dim:]
    #     for i, layer in enumerate(self.layers[:-1]):
    #         x = layer(x)
    #         # Optionally, insert batch normalization here if desired:
    #         # x = self.batch_norm[i](x)  # if you defined a BatchNorm1d layer per conv
            
    #         # Apply activation function and dropout
    #         x = self.do(self.nonlin(x))
    #         # Clamp values to a safe range (tweak the limits as needed)
    #         x = torch.clamp(x, min=-1e4, max=1e4)
    #         if torch.isnan(x).any() or torch.isinf(x).any():
    #             print(f"🚨 NaN/Inf detected after conv layer {i}")
    #             print("Layer weights:", layer.weight)
    #             break
    #     x = self.layers[-1](x)
    #     if torch.isnan(x).any() or torch.isinf(x).any():
    #         print("🚨 NaN/Inf detected after final conv layer")
    #     return x


    # def conv(self, x):
    #     x = x[:, x.shape[1] - self.input_dim:]
    #     for i, layer in enumerate(self.layers[:-1]):
    #         x = self.do(self.nonlin(layer(x)))
    #         if torch.isnan(x).any() or torch.isinf(x).any():
    #             print(f"🚨 NaN/Inf detected after conv layer {i}")
    #             print("Layer weights:", layer.weight)
    #             break
    #     x = self.layers[-1](x)
    #     if torch.isnan(x).any() or torch.isinf(x).any():
    #         print("🚨 NaN/Inf detected after final conv layer")
    #     return x



# def training_epoch(model, dataloader, optimizer):
#     """Train model for one epoch.
#     """
#     model.train()
#     loss_fn = nn.MSELoss()
#     train_loss = []
#     for x, y in tqdm(dataloader):
#         x = x.float().to(model.device)
#         y = y.float().to(model.device)
#         pred = model(x)
#         loss = loss_fn(pred, y)
#         train_loss += [loss.detach().cpu().numpy()]
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     train_loss = np.mean(train_loss)
#     return train_loss

def training_epoch(model, dataloader, optimizer):
    model.train()
    loss_fn = nn.MSELoss()
    train_loss = []
    for x, y in tqdm(dataloader):
        x = x.float().to(model.device)
        y = y.float().to(model.device)
        pred = model(x)
        loss = loss_fn(pred, y)
        # Check and clip gradients
        optimizer.zero_grad()
        loss.backward()
        # with torch.autograd.detect_anomaly():
        #     loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += [loss.detach().cpu().numpy()]
    return np.mean(train_loss)


# def compute_correlations(labels, preds, projects):
#     metrics = []
#     for project in np.unique(projects):
#         for i in range(labels.shape[1]):
#             y_true = labels[projects == project, i]
#             if len(np.unique(y_true)) > 1:
#                 y_prob = preds[projects == project, i]
#                 metrics.append(np.corrcoef(y_true, y_prob)[0, 1])
#     metrics = np.asarray(metrics)
#     return np.mean(metrics)

# def compute_correlations(labels, preds, projects):
#     metrics = []
#     for project in np.unique(projects):
#         for i in range(labels.shape[1]):
#             y_true = labels[projects == project, i]
#             # Skip or assign a default value if there is no variance.
#             if np.std(y_true) == 0:
#                 # Either skip this gene/project pair or count it as 0 correlation.
#                 metrics.append(0.0)
#             else:
#                 y_prob = preds[projects == project, i]
#                 corr = np.corrcoef(y_true, y_prob)[0, 1]
#                 # You might also check if the result is nan and then assign 0.
#                 if np.isnan(corr):
#                     corr = 0.0
#                 metrics.append(corr)
#     # If metrics remains empty for some reason, return 0 instead of nan.
#     return np.mean(metrics) if len(metrics) > 0 else 0.0

def compute_correlations(labels, preds, projects):
    metrics = []
    for project in np.unique(projects):
        for i in range(labels.shape[1]):
            y_true = labels[projects == project, i]
            if np.std(y_true) == 0:
                # Avoid computing correlation for constant values; or set default value like 0.
                metrics.append(0.0)
            else:
                y_prob = preds[projects == project, i]
                corr = np.corrcoef(y_true, y_prob)[0, 1]
                # Check in case corr returns NaN and assign default value
                if np.isnan(corr):
                    corr = 0.0
                metrics.append(corr)
    return np.mean(metrics) if metrics else 0.0


def evaluate(model, dataloader, projects):
    """Evaluate the model on the validation set and return loss and metrics.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    valid_loss = []
    preds = []
    labels = []
    for x, y in dataloader:
        pred = model(x.float().to(model.device))
        labels += [y]
        loss = loss_fn(pred, y.float().to(model.device))
        valid_loss += [loss.detach().cpu().numpy()]
        pred = nn.ReLU()(pred)
        preds += [pred.detach().cpu().numpy()]
    valid_loss = np.mean(valid_loss)
    if len(preds) == 0:
        print("Warning: No predictions were collected. Returning empty arrays.")
        return np.array([]), np.array([])
    else:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    # preds = np.concatenate(preds)
    # labels = np.concatenate(labels)
    metrics = compute_correlations(labels, preds, projects)
    return valid_loss, metrics


def predict(model, dataloader):
    """Perform prediction on the test set.
    """
    model.eval()
    labels = []
    preds = []
    for x, y in dataloader:
        pred = model(x.float().to(model.device))
        labels += [y]
        pred = nn.ReLU()(pred)
        preds += [pred.detach().cpu().numpy()]
    # preds = np.concatenate(preds)
    # labels = np.concatenate(labels)
    if len(preds) == 0:
        print("Warning: No predictions were collected. Returning empty arrays.")
        return np.array([]), np.array([])
    else:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    return preds, labels


def fit(model,
        train_set,
        valid_set,
        valid_projects,
        params={},
        optimizer=None,
        test_set=None,
        path=None,
        logdir='./exp'):
    """Fit the model and make prediction on evaluation set.

    Args:
        model (nn.Module)
        train_set (torch.utils.data.Dataset)
        valid_set (torch.utils.data.Dataset)
        valid_projects (np.array): list of integers encoding the projects
            validation samples belong to.
        params (dict): Dictionary for specifying training parameters.
            keys are 'max_epochs' (int, default=200), 'patience' (int,
            default=20) and 'batch_size' (int, default=16).
        optimizer (torch.optim.Optimizer): Optimizer for training the model
        test_set (None or torch.utils.data.Dataset): If None, return
            predictions on the validation set.
        path (str): Path to the folder where th model will be saved.
        logdir (str): Path for TensoboardX.
    """
    # torch.autograd.set_detect_anomaly(True)
    if path is not None and not os.path.exists(path):
        os.mkdir(path)

    default_params = {
        'max_epochs': 200,
        'patience': 20,
        'batch_size': 16,
        'num_workers': 0}
    default_params.update(params)
    batch_size = default_params['batch_size']
    patience = default_params['patience']
    max_epochs = default_params['max_epochs']
    num_workers = default_params['num_workers']

    writer = SummaryWriter(log_dir=logdir)
    
    # SET num_workers TO 0 WHEN WORKING WITH hdf5 FILES
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if valid_set is not None:
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if test_set is not None:
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if optimizer is None:
        # optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3,
        #                              weight_decay=0.)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.)

    metrics = 'correlations'
    epoch_since_best = 0
    start_time = time.time()

    if valid_set is not None:
        valid_loss, best = evaluate(
            model, valid_loader, valid_projects)
        print('{}: {:.3f}'.format(metrics, best))
        if np.isnan(best):
            best = 0
        if test_set is not None:
            preds, labels = predict(model, test_loader)
        else:
            preds, labels = predict(model, valid_loader)

    print("🔍 Checking training data for NaNs or Infs...")

    for i, (x, y) in enumerate(train_loader):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"❌ NaN or Inf found in input features at batch {i}")
            print("Sample tensor with issue (x):", x)
            break
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"❌ NaN or Inf found in labels at batch {i}")
            print("Sample tensor with issue (y):", y)
            break
    else:
        print("✅ Training data is clean. No NaNs or Infs found.")


    try:

        for e in range(max_epochs):

            epoch_since_best += 1

            train_loss = training_epoch(model, train_loader, optimizer)
            dic_loss = {'train_loss': train_loss}

            print('Epoch {}/{} - {:.2f}s'.format(
                e + 1,
                max_epochs,
                time.time() - start_time))
            start_time = time.time()

            if valid_set is not None:
                valid_loss, scores = evaluate(
                    model, valid_loader, valid_projects)
                dic_loss['valid_loss'] = valid_loss
                score = np.mean(scores)
                writer.add_scalars('data/losses',
                                   dic_loss,
                                   e)
                writer.add_scalar('data/metrics', score, e)
                print('loss: {:.4f}, val loss: {:.4f}'.format(
                    train_loss,
                    valid_loss))
                print('{}: {:.3f}'.format(metrics, score))
            else:
                writer.add_scalars('data/losses',
                                   dic_loss,
                                   e)
                print('loss: {:.4f}'.format(train_loss))

            if valid_set is not None:
                criterion = (score > best)

                if criterion:
                    epoch_since_best = 0
                    best = score
                    if path is not None:
                        torch.save(model, os.path.join(path, 'model.pt'))
                    elif test_set is not None:
                        preds, labels = predict(model, test_loader)
                    else:
                        preds, labels = predict(model, valid_loader)

                if epoch_since_best == patience:
                    print('Early stopping at epoch {}'.format(e + 1))
                    break

    except KeyboardInterrupt:
        pass
    
    

    if path is not None and os.path.exists(os.path.join(path, 'model.pt')):
        from model import HE2RNA  # ensure HE2RNA is imported
        from torch.nn.modules.conv import Conv1d 
        torch.serialization.add_safe_globals([HE2RNA, Conv1d])
        model = torch.load(os.path.join(path, 'model.pt'), weights_only=False)

    elif path is not None:
        torch.save(model, os.path.join(path, 'model.pt'))

    if test_set is not None:
        preds, labels = predict(model, test_loader)
    elif valid_set is not None:
        preds, labels = predict(model, valid_loader)
    else:
        preds = None
        labels = None

    writer.close()

    return preds, labels
