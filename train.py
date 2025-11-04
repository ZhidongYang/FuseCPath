import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from regularization import Regularization

from my_dataset_classification_collaborated_kd import MyDataSet as WSI_Dataset
from vit_model_collaborated_kd import rrt_mil_col_kd_directconcat as create_model
from utils_collaborated_kd import train_one_epoch_mvckd_collaborated, evaluate_mvckd_collaborated, fix_random_seeds
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.model_output_path) is False:
        os.makedirs(args.model_output_path)

    tb_writer = SummaryWriter()

    # instantiate training data 
    train_dataset = WSI_Dataset(result_txt_path=args.wsi_train_feat_dir,
                                label_path=args.label_path,
                                teacher_emds_path_1=args.teacher_emds_path_1,
                                teacher_emds_path_2=args.teacher_emds_path_2,
                                teacher_emds_path_3=args.teacher_emds_path_3,
                              mode='train')
    # exit(0)
    # instantiate validation data 
    val_dataset = WSI_Dataset(result_txt_path=args.wsi_valid_feat_dir,
                                label_path=args.label_path,
                                teacher_emds_path_1=args.teacher_emds_path_1,
                                teacher_emds_path_2=args.teacher_emds_path_2,
                                teacher_emds_path_3=args.teacher_emds_path_3,
                              mode='valid')

    # exit(0)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = 1  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=2).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    if args.weight_decay > 0:
        regular_loss = Regularization(model, args.weight_decay, p=1).to(device)
    else:
        regular_loss = None
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # optimizer = optim.Adam(pg, lr=args.lr, weight_decay=1E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch_mvckd_collaborated(model=model,
                                                alpha=args.alpha,
                                                T=args.temperature,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate_mvckd_collaborated(model=model,
                                     alpha=args.alpha,
                                     T=args.temperature,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), args.model_output_path+"/model-luad_classfication_ep{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')

    # directories
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/hdd_data/yangzd/flower_photos")
    parser.add_argument('--model_output_path', type=str,
                        default='./weights_20250608_luad_rrtmil_kd_3wsiFM_egfr')
    parser.add_argument('--wsi_train_feat_dir', type=str,
                        default='/hdd_data/yangzd/prov-gigapath/gigapath_features_training/train_select_feat_extended')
    parser.add_argument('--wsi_valid_feat_dir', type=str,
                        default='/hdd_data/yangzd/prov-gigapath/gigapath_features_training/valid_select_feat_extended')
    parser.add_argument('--label_path', type=str,
                        default='/hdd_data/yangzd/TCGA_LUAD_mutation_labels/LUAD_TP53_mutation_list.txt')
    parser.add_argument('--teacher_emds_path_1', type=str,
                        default='/data3/ruiyan/yzd/trident/trident_features/20x_256px_0px_overlap_gigapath/slide_features_gigapath')
    parser.add_argument('--teacher_emds_path_2', type=str,
                        default='/data3/ruiyan/yzd/trident/trident_features/20x_512px_0px_overlap/slide_features_titan')
    parser.add_argument('--teacher_emds_path_3', type=str,
                        default='/data3/ruiyan/yzd/trident/trident_features/20x_224px_0px_overlap/slide_features_prism')
    parser.add_argument('--model-name', default='', help='create model name')

    # whether use pre-trained weights
    parser.add_argument('--weights', type=str, default='', help='initial weights path')

    # freeze layers
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    fix_random_seeds(seed=opt.seed)
    print(opt)
    # exit(0)

    main(opt)
