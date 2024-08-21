import torch
import torch.nn as nn


def model_train(model, train_loader, optm, device):
    train_loss = 0
    criteria = nn.MSELoss()

    for idx, (feature, label) in enumerate(train_loader):
        loss = loss_calculation(feature, label, model, criteria, device)

        loss.backward()
        optm.step()
        optm.zero_grad()

        train_loss += loss.detach()

    return train_loss / len(train_loader)


def model_test(model, test_loader, device):
    model.eval()

    test_loss = 0
    criteria = nn.MSELoss()

    for idx, (feature, label) in enumerate(test_loader):
        loss = loss_calculation(feature, label, model, criteria, device)

        test_loss += loss.detach()
    return test_loss / len(test_loader)


def loss_calculation(feature, label, model, criteria, device):
    loss = criteria(torch.zeros(1, 2), torch.zeros(1, 2)).to(device)

    # Get the data
    x = feature.to(device)
    x_next = label.to(device)
    x_next = x_next.reshape(64, 3, 10)

    # Reconstruction
    y = model.observe(x)
    x_rec = model.recover(y)
    loss += criteria(x, x_rec)

    # State prediction
    for i in range(10):
        y = model.predict(y)
        x_pred = model.recover(y)
        loss += (1/10) * criteria(x_next[:, :, i], x_pred)

    # Koopman observable
    y_pred = model.observe(x)
    for i in range(10):
        y_pred = model.predict(y_pred)
        y_obs = model.observe(x_next[:, :, i])
        loss += (1/10) * criteria(y_obs, y_pred)

    #k_matrix_raw = model.k_matrix_raw
    #k_matrix = k_matrix_raw.view(model.output_dim, model.output_dim)
    #loss += torch.norm(k_matrix - torch.diag_embed(torch.diagonal(k_matrix)), p=2)

    return loss
