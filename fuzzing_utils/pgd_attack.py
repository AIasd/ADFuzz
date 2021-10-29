import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
import torch.nn.functional as F
from torchvision import models
from customized_utils import (
    if_violate_constraints,
    customized_standardize,
    customized_inverse_standardize,
    recover_fields_not_changing,
    decode_fields,
    is_distinct_vectorized
)


class VanillaDataset(Data.Dataset):
    def __init__(self, X, y, one_hot=False, to_tensor=False):
        self.X = X
        self.y = y
        self.one_hot = one_hot
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.to_tensor:
            return (
                torch.from_numpy(np.array(self.X[idx])),
                torch.from_numpy(np.array(self.y[idx])),
            )
        else:
            return (self.X[idx], self.y[idx])


class BNN(nn.Module):
    def __init__(self, input_size, output_size, device=None):
        super(BNN, self).__init__()

        # self.layers = nn.Sequential(
        #     nn.Linear(input_size, 150),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     # nn.Linear(20, 20),
        #     # nn.Dropout(0.5),
        #     # nn.ReLU(),
        #     nn.Linear(150, output_size),
        #     nn.Sigmoid())

        self.fc1 = nn.Linear(input_size, 150)
        self.dropout = nn.Dropout(0.5)
        self.relu1 = nn.ReLU()
        self.fc_end = nn.Linear(150, output_size)
        self.sigmoid = nn.Sigmoid()

        if not device:
            self.device = torch.device("cuda")
        else:
            self.device = device

    def forward(self, x, return_logits=False):
        # x = torch.flatten(x, 1)
        # logits = self.layers(x.float())

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu1(x)
        x = self.fc_end(x)
        logits = self.sigmoid(x)

        return logits

    def predict(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)
        out = torch.round(out)
        return out.cpu().detach().numpy()

    def predict_proba(self, x):
        if isinstance(x, np.ndarray):
            is_numpy = True
        else:
            is_numpy = False
        if is_numpy:
            x = torch.from_numpy(x).to(self.device).float()

        out = self.forward(x)
        out = torch.stack([1 - out, out], dim=1).squeeze()
        # print(out.cpu().detach().numpy().shape)

        if is_numpy:
            return out.cpu().detach().numpy()
        else:
            return out


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device=None):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc_end = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

        if not device:
            self.device = torch.device("cuda")
        else:
            self.device = device

    def extract_embed(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        return out.cpu().detach().numpy()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc_end(out)
        out = self.sigmoid(out)
        return out

    def predict(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)
        out = torch.round(out)
        return out.cpu().detach().numpy()

    def predict_proba(self, x):
        if isinstance(x, np.ndarray):
            is_numpy = True
        else:
            is_numpy = False
        if is_numpy:
            x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)

        out = torch.stack([1 - out, out], dim=1).squeeze()
        # print(out.cpu().detach().numpy().shape)
        if is_numpy:
            return out.cpu().detach().numpy()
        else:
            return out


# class SimpleNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, device=None):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.tanh = nn.Tanh()
#         self.fc_last = nn.Linear(hidden_size, num_classes)
#         self.sigmoid = nn.Sigmoid()
#
#         if not device:
#             self.device = torch.device("cuda")
#         else:
#             self.device = device
#     def extract_embed(self, x):
#         x = torch.from_numpy(x).to(self.device).float()
#         out = self.fc1(x)
#         out = self.tanh(out)
#         return out.cpu().detach().numpy()
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.tanh(out)
#         out = self.fc_last(out)
#         out = self.sigmoid(out)
#         return out
#     def predict(self, x):
#         x = torch.from_numpy(x).to(self.device).float()
#         out = self.forward(x)
#         out = torch.round(out)
#         return out.cpu().detach().numpy()
#     def predict_proba(self, x):
#         x = torch.from_numpy(x).to(self.device).float()
#         out = self.forward(x)
#
#         out = torch.stack([1 - out, out], dim=1).squeeze()
#         # print(out.cpu().detach().numpy().shape)
#         return out.cpu().detach().numpy()


class SimpleNetMulti(SimpleNet):
    def predict(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)
        out = torch.argmax(out)
        return out.cpu().detach().numpy()

    def predict_proba(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)
        return out.cpu().detach().numpy()


class SimpleRegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device=None):
        super(SimpleRegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        if not device:
            self.device = torch.device("cuda")
        else:
            self.device = device

    def extract_embed(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        return out

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)
        return out.cpu().detach().numpy()

    def predict_proba(self, x):
        if isinstance(x, np.ndarray):
            is_numpy = True
        else:
            is_numpy = False
        if is_numpy:
            x = torch.from_numpy(x).to(self.device).float()
        out = self.forward(x)

        out = torch.stack([out, 20-out], dim=1).squeeze()
        # print(out.cpu().detach().numpy().shape)
        if is_numpy:
            return out.cpu().detach().numpy()
        else:
            return out

def extract_embed(model, X):
    X_torch = torch.from_numpy(X).cuda().float()
    output = model.extract_embed(X_torch)
    return output.cpu().detach().numpy()


def validation(model, test_loader, device, one_hot=True, regression=False):
    mean_loss = []
    mean_acc = []
    model.eval()
    if regression:
        one_hot = False
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device).float()
        y_batch = y_batch.to(device).float()

        if one_hot:
            y_batch = y_batch.long()
        y_pred_batch = model(x_batch).squeeze()
        loss = criterion(y_pred_batch, y_batch)
        diff_np = torch.abs(y_pred_batch - y_batch).cpu().detach().numpy()
        loss_np = loss.cpu().detach().numpy()

        y_batch_np = y_batch.cpu().detach().numpy()
        y_pred_batch_np = y_pred_batch.cpu().detach().numpy()

        acc = np.mean(np.round(y_pred_batch_np) == y_batch_np)

        mean_loss.append(loss_np)
        mean_acc.append(acc)
        # print('test', y_pred_batch, y_batch, loss_np, acc)

    mean_loss = np.mean(mean_loss)
    mean_acc = np.mean(mean_acc)

    return mean_loss, mean_acc, diff_np


def pgd_attack(
    model,
    images,
    labels,
    xl,
    xu,
    encoded_fields,
    labels_used,
    customized_constraints,
    standardize,
    prev_X=[],
    base_ind=0,
    unique_coeff=None,
    mask=None,
    param_for_recover_and_decode=None,
    device=None,
    eps=1.01,
    adv_conf_th=0,
    attack_stop_conf=1,
    alpha=1 / 255,
    iters=255,
    max_projections_steps=3,
    associated_clf_id=[],
    X_test_pgd_ori=[],
    consider_uniqueness=False
):
    if len(associated_clf_id) > 0:
        print(len(model))
        print(associated_clf_id)
        multiple_models = True
    else:
        multiple_models = False

    if not device:
        device = torch.device("cuda")
    n = len(images)
    encoded_fields_len = np.sum(encoded_fields)

    images_all = torch.from_numpy(images).to(device).float()
    labels_all = torch.from_numpy(labels).to(device).float()
    ori_images_all = torch.clone(images_all)

    xl = torch.from_numpy(xl).to(device).float()
    xu = torch.from_numpy(xu).to(device).float()

    loss = nn.BCELoss()

    new_images_all = []
    new_outputs_all = []
    prev_x_all = []
    initial_outputs_all = []

    if consider_uniqueness:
        (
            X_removed,
            kept_fields,
            removed_fields,
            enc,
            inds_to_encode,
            inds_non_encode,
            encoded_fields,
            xl_ori,
            xu_ori,
            unique_bugs_len,
        ) = param_for_recover_and_decode
        p, c, th = unique_coeff

    if_low_conf_examples = np.zeros(n)

    # we deal with images sequentially
    for j in range(n):
        images = torch.unsqueeze(images_all[j], 0)
        labels = labels_all[j]
        ori_images = torch.unsqueeze(ori_images_all[j], 0)
        ori_images_encode = ori_images[:, :encoded_fields_len]
        ori_images_non_encode = ori_images[:, encoded_fields_len:]



        prev_outputs = torch.zeros(1).to(device).float()
        prev_images = []
        current_x = []
        prev_x = []

        max_violate_times = 10
        violate_times = 0

        if multiple_models:
            model_id = associated_clf_id[j]
            cur_model = model[model_id]
            print("adv_conf_th", adv_conf_th)
            if type(adv_conf_th) == type([]) and len(adv_conf_th) > 0:
                cur_adv_conf_th = adv_conf_th[model_id]
                print("cur_adv_conf_th 1", cur_adv_conf_th)
            else:
                cur_adv_conf_th = adv_conf_th
                print("cur_adv_conf_th 2", cur_adv_conf_th)
            if type(attack_stop_conf) == type([]) and len(attack_stop_conf) > 0:
                cur_attack_stop_conf = attack_stop_conf[model_id]
            else:
                cur_attack_stop_conf = attack_stop_conf

            print("model_id", model_id)
        else:
            cur_model = model
            cur_adv_conf_th = adv_conf_th
            cur_attack_stop_conf = attack_stop_conf
        for i in range(iters):

            images.requires_grad = True
            outputs = cur_model(images).squeeze()
            cur_model.zero_grad()

            cost = loss(outputs, labels).to(device)
            cost.backward()

            outputs_np = outputs.squeeze().cpu().detach().numpy()
            # print('\n'*2)
            # print(i, outputs_np)
            # print('\n'*2)
            if i == 0:
                initial_outputs_all.append(outputs_np)
                print("\n", j, "initial outputs", outputs_np, "\n")

            # check uniqueness of new x

            distinct = True
            if consider_uniqueness:
                ind = base_ind + j
                current_x = images.squeeze().cpu().detach().numpy()
                current_x = customized_inverse_standardize(
                    np.array([current_x]), standardize, encoded_fields_len, True
                )[0]
                current_x = recover_fields_not_changing(
                    np.array([current_x]),
                    np.array(X_removed[ind]),
                    kept_fields,
                    removed_fields,
                )[0]
                current_x = decode_fields(
                    np.array([current_x]),
                    enc,
                    inds_to_encode,
                    inds_non_encode,
                    encoded_fields,
                    adv=True,
                )[0]

                if len(prev_x_all) > 0:
                    prev_X_and_prev_x_all = np.concatenate([prev_X, prev_x_all])
                else:
                    prev_X_and_prev_x_all = prev_X

                if len(X_test_pgd_ori) > 1 and j < n-1:
                    prev_X_and_prev_x_all = np.concatenate([prev_X_and_prev_x_all, X_test_pgd_ori[j+1:]])

                remaining_inds = is_distinct_vectorized(
                    [current_x], prev_X_and_prev_x_all, mask, xl_ori, xu_ori, p, c, th, verbose=False
                )
                if len(remaining_inds) == 1:
                    distinct = True
                else:
                    distinct = False


            # if new x is close to previous X or forward prob not improving, break
            cond1 = not distinct and i > 0
            cond2 = (outputs - prev_outputs) < 1e-3
            cond4 = (
                i > 0 and prev_outputs.cpu().detach().numpy() >= cur_attack_stop_conf
            )
            # print('prev_outputs.cpu().detach().numpy()', prev_outputs.cpu().detach().numpy())
            if cond1 or cond2 or cond4:
                if cond1:
                    print("cond1 with step", i)
                elif cond2:
                    print("cond2 with step", i)
                elif cond4:
                    print("cond4 with step", i)
                break
            else:
                # print('update x with the current one')
                prev_images = torch.clone(images)
                prev_outputs = torch.clone(outputs)
                prev_x = current_x
                if i == 0 and prev_outputs.cpu().detach().numpy() > cur_adv_conf_th:
                    print("cond3 with step", i)
                    if_low_conf_examples[j] = 1
                    print(
                        "num_of_high_conf_examples",
                        np.sum(if_low_conf_examples),
                        "/",
                        j + 1,
                    )
                    break

            adv_images = images + alpha * images.grad.sign()
            # print('images.grad', images.grad.cpu().detach().numpy(), '\n'*2)
            eta = adv_images - ori_images

            # print('ori_images', ori_images.cpu().detach().numpy(), '\n'*2)
            # print('\n'*2, 'eta', eta.cpu().detach().numpy(), '\n'*2)
            # eta[:, :encoded_fields_len] = torch.clip(eta[:, :encoded_fields_len], min=-eps, max=eps)
            # print('eps', eps)
            # print('\n'*2, 'eta clipped', eta.cpu().detach().numpy(), '\n'*2)
            eta = torch.clip(eta, min=-eps, max=eps)
            # print('\n'*2, 'eta clipped 2', eta.cpu().detach().numpy(), '\n'*2)
            # print('\n'*2, 'xl', xl.cpu().detach().numpy(), '\n'*2)
            # print('\n'*2, 'xu', xu.cpu().detach().numpy(), '\n'*2)

            eta = eta * (xu - xl)
            # print('\n'*2, 'eta * (xu - xl)', eta.cpu().detach().numpy(), '\n'*2)
            images = torch.max(torch.min(ori_images + eta, xu), xl).detach_()

            one_hotezed_images_embed = torch.zeros(
                [images.shape[0], encoded_fields_len]
            )
            s = 0

            for field_len in encoded_fields:
                max_inds = torch.argmax(images[:, s : s + field_len], axis=1)
                one_hotezed_images_embed[
                    torch.arange(images.shape[0]), s + max_inds
                ] = 1
                # print(images.cpu().detach().numpy())
                # print(field_len, max_inds.cpu().detach().numpy())
                # print(one_hotezed_images_embed.cpu().detach().numpy())
                s += field_len
            images[:, :encoded_fields_len] = one_hotezed_images_embed

            images_non_encode = images[:, encoded_fields_len:]
            images_delta_non_encode = images_non_encode - ori_images_non_encode
            xl_non_encode_np = xl[encoded_fields_len:].squeeze().cpu().numpy()
            xu_non_encode_np = xu[encoded_fields_len:].squeeze().cpu().numpy()

            # keep checking violation, exit only when satisfying
            ever_violate = False

            images_non_encode_np = images_non_encode.squeeze().cpu().numpy()
            ori_images_non_encode_np = ori_images_non_encode.squeeze().cpu().numpy()
            images_delta_non_encode_np = images_delta_non_encode.squeeze().cpu().numpy()

            satisfy_constraints = False
            for k in range(max_projections_steps):
                # print('images_non_encode_np', images_non_encode_np.shape)
                images_non_encode_np_inv_std = customized_inverse_standardize(
                    np.array([images_non_encode_np]),
                    standardize,
                    encoded_fields_len,
                    False,
                )[0]
                if_violate, [
                    violated_constraints,
                    involved_labels,
                ] = if_violate_constraints(
                    images_non_encode_np_inv_std,
                    customized_constraints,
                    labels_used,
                    verbose=False,
                )
                # if violate, pick violated constraints, project perturbation back to linear constraints via LR
                if if_violate:
                    ever_violate = True
                    # print(len(images_delta_non_encode_np), m)
                    # print(images_delta_non_encode_np)
                    images_delta_non_encode_np_inv_std = customized_inverse_standardize(
                        np.array([images_delta_non_encode_np]),
                        standardize,
                        encoded_fields_len,
                        False,
                        True,
                    )

                    new_images_delta_non_encode_np_inv_std = project_into_constraints(
                        images_delta_non_encode_np_inv_std[0],
                        violated_constraints,
                        labels_used,
                        involved_labels,
                    )

                    # print(ori_images.squeeze().cpu().numpy())
                    # print(images_delta_non_encode_np_inv_std[0])
                    # print(new_images_delta_non_encode_np_inv_std)
                else:
                    satisfy_constraints = True
                    break

                new_images_delta_non_encode_np = customized_standardize(
                    np.array([new_images_delta_non_encode_np_inv_std]),
                    standardize,
                    encoded_fields_len,
                    False,
                    True,
                )[0]

                # print(new_images_delta_non_encode_np.shape, new_images_delta_non_encode_np.shape)

                images_non_encode_np = (
                    ori_images_non_encode_np + new_images_delta_non_encode_np
                )

                # print('-- check violation before clip')
                # images_non_encode_np_inv_std_tmp = customized_inverse_standardize(np.array([images_non_encode_np]), standardize, m, False)[0]
                # _, _ = if_violate_constraints(images_non_encode_np_inv_std_tmp, customized_constraints, labels_used, verbose=True)
                # print(images_non_encode_np_inv_std_tmp)
                # print('++ check violation before clip')

                eta = np.clip(
                    images_non_encode_np - ori_images_non_encode_np, -eps, eps
                )
                # eta *= (1/(violate_times+1))
                images_non_encode_np = np.maximum(
                    np.minimum(ori_images_non_encode_np + eta, xu_non_encode_np),
                    xl_non_encode_np,
                )

                # print('-- check violation after clip')
                images_non_encode_np_inv_std_tmp = customized_inverse_standardize(
                    np.array([images_non_encode_np]),
                    standardize,
                    encoded_fields_len,
                    False,
                )[0]
                if_violate_after_clip, _ = if_violate_constraints(
                    images_non_encode_np_inv_std_tmp,
                    customized_constraints,
                    labels_used,
                    verbose=False,
                )
                # print(images_non_encode_np_inv_std_tmp)
                # print('++ check violation after clip')

                if if_violate_after_clip:
                    satisfy_constraints = False

                # print(ori_images_non_encode_np)
                # print(new_images_delta_non_encode_np)
                # print(images_non_encode_np)

                # ori_images_non_encode_np_inv_std = customized_inverse_standardize(np.array([ori_images_non_encode_np]), standardize, m, False)[0]
                # images_non_encode_np_inv_std = customized_inverse_standardize(np.array([images_non_encode_np]), standardize, m, False)[0]
                # print(standardize.mean_, standardize.scale_)
                # print(ori_images_non_encode_np_inv_std)
                # print(new_images_delta_non_encode_np_inv_std)
                # print(images_non_encode_np_inv_std)

            if not satisfy_constraints or violate_times > max_violate_times:
                break
            if ever_violate:
                violate_times += 1
                # print('ever_violate', ever_violate, m)
                images_non_encode = torch.from_numpy(images_non_encode_np).to(device)
                images[:, encoded_fields_len:] = images_non_encode

            # if i == iters - 1:
            #     print('iter', i, ':', 'cost :', cost.cpu().detach().numpy(), 'outputs :', outputs.cpu().detach().numpy())

        print(
            "\n", "final outputs", prev_outputs.squeeze().cpu().detach().numpy(), "\n"
        )
        if len(prev_images) > 0:
            new_images_all.append(prev_images.squeeze().cpu().detach().numpy())
            new_outputs_all.append(prev_outputs.squeeze().cpu().detach().numpy())
            prev_x_all.append(prev_x)

    print("\n" * 2)
    print("num_of_high_conf_examples", np.sum(if_low_conf_examples), "/", n)
    print("\n" * 2)
    print(if_low_conf_examples)
    print(np.array(new_outputs_all))

    return (
        np.array(new_images_all),
        np.array(new_outputs_all),
        np.array(initial_outputs_all),
    )


def train_net(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_train=64,
    batch_test=20,
    hidden_size=150,
    model_type="one_output",
    device=None,
    num_epochs=30
):
    if not device:
        device = torch.device("cuda")
    input_size = X_train.shape[1]


    if model_type == "one_output":
        num_classes = 1
        model = SimpleNet(input_size, hidden_size, num_classes)
        criterion = nn.BCELoss()
        one_hot = False
    elif model_type == "BNN":
        num_classes = 1
        model = BNN(input_size, num_classes)
        criterion = nn.BCELoss()
        one_hot = False
    elif model_type == "two_output":
        num_classes = 2
        model = SimpleNetMulti(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        one_hot = True
    elif model_type == "regression":
        num_classes = 1
        model = SimpleRegressionNet(input_size, hidden_size, num_classes)
        criterion = nn.MSELoss()
        one_hot = False
    else:
        raise "unknown model_type " + model_type

    model.cuda()

    # optimizer = torch.optim.LBFGS(model.parameters())
    optimizer = torch.optim.Adam(model.parameters())

    d_train = VanillaDataset(X_train, y_train)

    train_loader = Data.DataLoader(d_train, batch_size=batch_train, shuffle=True)

    # class_sample_count = [np.sum(y_train==0), np.sum(y_train==1)]
    # class_sample_count = [y_train.shape[0]/2, y_train.shape[0]/2]
    # weights = 1 / torch.Tensor(class_sample_count)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_train)
    # train_loader = Data.DataLoader(d_train, batch_size=batch_train, sampler=sampler)

    if len(y_test) > 0:
        d_test = VanillaDataset(X_test, y_test)
        test_loader = Data.DataLoader(d_test, batch_size=batch_test, shuffle=True)

    # Train the Model
    counter = 0
    for epoch in range(num_epochs):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            if one_hot:
                y_batch = y_batch.long()

            optimizer.zero_grad()
            y_pred_batch = model(x_batch).squeeze(dim=1)

            loss = criterion(y_pred_batch, y_batch)
            loss.backward()
            optimizer.step()

            counter += 1
            if epoch % 1 == 0 and len(y_test) > 0:
                mean_loss, mean_acc, _ = validation(model, test_loader, device, one_hot)
                print(
                    "Epoch [%d/%d], Step %d, Test Mean Loss: %.4f, Test Mean Accuracy: %.4f"
                    % (epoch + 1, num_epochs, counter, mean_loss, mean_acc)
                )
                model.train()

    return model


def train_regression_net(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_train=200,
    batch_test=20,
    device=None,
    hidden_layer_size=100,
    return_test_err=False,
):
    if not device:
        device = torch.device("cuda")
    input_size = X_train.shape[1]
    hidden_size = hidden_layer_size
    num_epochs = 200

    num_classes = 1
    model = SimpleRegressionNet(input_size, hidden_size, num_classes)
    criterion = nn.MSELoss()

    model.cuda()

    # optimizer = torch.optim.LBFGS(model.parameters())
    optimizer = torch.optim.Adam(model.parameters())

    d_train = VanillaDataset(X_train, y_train)

    # class_sample_count = [np.sum(y_train==0), np.sum(y_train==1)]
    # class_sample_count = [y_train.shape[0]/2, y_train.shape[0]/2]
    # weights = 1 / torch.Tensor(class_sample_count)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_train)
    # train_loader = Data.DataLoader(d_train, batch_size=batch_train, sampler=sampler)

    train_loader = Data.DataLoader(d_train, batch_size=batch_train, shuffle=True)

    if len(y_test) > 0:
        d_test = VanillaDataset(X_test, y_test)
        test_loader = Data.DataLoader(d_test, batch_size=batch_test, shuffle=True)

    # Train the Model
    counter = 0
    for epoch in range(num_epochs):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()
            y_pred_batch = model(x_batch).squeeze()

            loss = criterion(y_pred_batch, y_batch)
            loss.backward()
            optimizer.step()

            counter += 1
            # if epoch % 1 == 0 and len(y_test) > 0:
            #     mean_loss, _ = validation(model, test_loader, device, regression=True)
            #     print ('Epoch [%d/%d], Step %d, Test Mean Loss: %.4f'
            #            %(epoch+1, num_epochs, counter, mean_loss))
            #     model.train()
    if return_test_err:
        if len(y_test) > 0:
            _, _, diff_np = validation(model, test_loader, device, regression=True)
            conf = np.percentile(diff_np, 95)
        else:
            conf = 0

    if return_test_err:
        return model, conf
    else:
        return model


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, weights):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weights)

    def forward(self, x):
        out = self.linear(x)
        return out


def project_into_constraints(x, violated_constraints, labels, involved_labels):
    assert len(labels) == len(x), str(len(labels)) + " VS " + str(len(x))
    labels_to_id = {label: i for i, label in enumerate(labels)}
    # print(labels_to_id)
    # print(involved_labels)
    involved_ids = np.array([labels_to_id[label] for label in involved_labels])
    map_ids = {involved_id: i for i, involved_id in enumerate(involved_ids)}

    m = len(violated_constraints)
    r = len(involved_ids)
    A_train = np.zeros((m, r))
    x_new = x.copy()
    # print('involved_ids', involved_ids)
    x_start = x[involved_ids]
    y_train = np.zeros(m)

    for i, constraint in enumerate(violated_constraints):
        ids = np.array([map_ids[labels_to_id[label]] for label in constraint["labels"]])
        A_train[i, ids] = np.array(constraint["coefficients"])
        y_train[i] = constraint["value"]

    x_projected = LR(A_train, x_start, y_train)
    x_new[involved_ids] = x_projected
    return x_new


def LR(A_train, x_start, y_train):
    # A_train ~ m * r, constraints
    # x_start ~ r * 1, initial x
    # y_train ~ m * 1, target values
    # m = constraints number
    # r = number of variables involved
    # print('x_start.shape', x_start.shape)
    x_start = torch.from_numpy(x_start).cuda().float()

    inputDim = A_train.shape[1]
    outputDim = 1
    learningRate = 3e-4
    epochs = 300
    eps = 1e-7
    model = linearRegression(inputDim, outputDim, x_start)
    model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        # print('A_train.shape', A_train.shape)

        inputs = torch.from_numpy(A_train).cuda().float()
        labels = torch.from_numpy(y_train).cuda().float().unsqueeze(0)

        optimizer.zero_grad()
        outputs = model(inputs)
        # print('outputs.size(), labels.size()', outputs.size(), labels.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print('epoch {}, loss {}'.format(epoch, loss.item()))
        if loss.item() < eps:
            break

    return model.linear.weight.data.cpu().numpy()
