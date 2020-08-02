import torch
from torchvision import transforms

from pytorch.segmentation.data_loader.segmentation_dataset import SegmentationDataset
from pytorch.segmentation.data_loader.transform import Rescale, ToTensor
from pytorch.segmentation.trainer import Trainer
from pytorch.segmentation.models import all_models
from pytorch.util.logger import Logger

# train_images = r'D:/datasets/cityspaces_full/images/train'
# test_images = r'D:/datasets/cityspaces_full/images/test'
# train_labled = r'D:/datasets/cityspaces_full/labeled/train'
# test_labeled = r'D:/datasets/cityspaces_full/labeled/test'
train_images = r'D:/datasets/cityspaces/images/train/aachen'
test_images = r'D:/datasets/cityspaces/images/test/lindau'
train_labled = r'D:/datasets/cityspaces/labeled/train/aachen'
test_labeled = r'D:/datasets/cityspaces/labeled/test/lindau'

if __name__ == '__main__':
    #model_name = "pspnet_mobilenet_v2"
    device = 'cuda'
    batch_size = 1
    n_classes = 34
    input_axis_minimum_size = 500
    fixed_feature = False
    pretrained = True
    num_epochs = 300
    check_point_stride = 1

    logger = Logger(model_name=model_name, data_name='example')

    # Loader
    compose = transforms.Compose([
        Rescale(input_axis_minimum_size),
        ToTensor()
         ])

    train_datasets = SegmentationDataset(train_images, train_labled, n_classes, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    #test_datasets = SegmentationDataset(test_images, test_labeled, n_classes, compose)
    #test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = None

    # Model
    batch_norm = False if batch_size == 1 else True
    model = all_models.model_from_name[model_name](n_classes,
                                                   batch_norm=batch_norm,
                                                   pretrained=pretrained,
                                                   fixed_feature=fixed_feature)
    model.to(device)
    #logger.load_models(model, 'epoch123')
    # Optimizers

    if pretrained and fixed_feature: #fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    # Train
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(model, optimizer, logger, num_epochs,
                      train_loader, test_loader, epoch=0, check_point_epoch_stride=check_point_stride)
    trainer.train()


