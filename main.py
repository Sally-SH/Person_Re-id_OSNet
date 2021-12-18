from data.datamanager import ImageDataManager
from osnet import osnet
from utils.lr_scheduler import build_lr_scheduler
from utils.torchtools import load_pretrained_weights
from softmax import ImageSoftmaxEngine
import torch

# load dataset
datamanager = ImageDataManager(
    root='dataset',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop'],
    norm_mean=[0.4154, 0.3897, 0.3849],
    norm_std=[0.1930, 0.1865, 0.1850]
)

# build OSNet
model = osnet(
    num_classes=datamanager.num_train_pids,
    loss='softmax',
)
model = model.cuda()

#load_pretrained_weights(model,"./log/osnet/model/model.pth.tar-60")

# build optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003,
    weight_decay=5e-04,
    betas=(0.9, 0.99),
)

#function for decay learning rate
scheduler = build_lr_scheduler(
    optimizer,
    lr_scheduler='multi_step',
    stepsize=[30,50]
)

engine = ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir='log/osnet',
    max_epoch=60,
    eval_freq=10,
    print_freq=20,
    test_only=False,
    visrank=False,
    open_layers='classifier'
)
