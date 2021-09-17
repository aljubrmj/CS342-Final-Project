import torch
import numpy as np

from models import FCN, save_model
from utils import load_regression_data, ConfusionMatrix
import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code

    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = FCN().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.MSELoss()

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_dense_data('data_2/train', num_workers=4, transform=transform)
    valid_data = load_dense_data('data_2/valid', num_workers=4)

    global_step = 0
    best_valid_mse = 1000
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device).long()

            pred = model(img)
            loss_val = loss(pred, label).mean()
            loss_vals.append(loss_val)
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        mean_train_loss = torch.mean(torch.stack(loss_vals))
        model.eval()
        loss_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device).long()
            pred = model(img)
            loss_val = loss(pred, label).mean()
            loss_vals.append(loss_val)
        
        mean_valid_loss = torch.mean(torch.stack(loss_vals))
        if valid_logger is not None:
                valid_logger.add_scalar('loss', loss_val, global_step)
        if valid_logger is not None:
            log(valid_logger, img, label, logit, global_step)

        print(f"epoch {epoch}:   train_MSE: {mean_train_loss:.2f}   valid_MSE: {mean_train_loss:.2f}")
        
        if mean_valid_loss < best_valid_mse:
            best_valid_mse = mean_valid_loss
            print('Saving Model.')
            save_model(model)

    def log(logger, img, label, pred, global_step):
        import matplotlib.pyplot as plt
        import torchvision.transforms.functional as TF
        fig, ax = plt.subplots(1, 1)
        ax.imshow(TF.to_pil_image(img[0].cpu()))
        WH2 = np.array([img.size(-1), img.size(-2)])/2
        ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
        ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
        logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
