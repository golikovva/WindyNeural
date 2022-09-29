model_name = "spice_self"
pre_model = "./results/sediments/moco/checkpoint_final.pth.tar"
embedding = "./results/sediments/embedding/feas_moco_512_l2.npy"
resume = "./results/sediments/{}/checkpoint_last.pth.tar".format(model_name)
model_type = "clusterresnet"
num_workers = 16
device_id = 0
num_train = 5
num_cluster = 10
batch_size = 32
target_sub_batch_size = 100
train_sub_batch_size = 128
batch_size_test = 100
num_trans_aug = 1
num_repeat = 8


epochs = 100


start_epoch = 0
print_freq = 1
test_freq = 1

model = dict(
    new_in_channels=14,
    gfs_input_size=512,
    gfs_hidden_size=512,
    num_layers=1,
    station_input_size=3,
    forecast_range=6,
    pretrained=pre_model,
    freeze_conv=True,
)

data_train = dict(
    type="sediments_emb",
    root_folder="./datasets/sediments",
    embedding=embedding,
    split="train+test",
    ims_per_batch=batch_size,
    shuffle=True,
    aspect_ratio_grouping=False,
    train=True,
    show=False,
    trans1=dict(
        aug_type="weak",
        crop_size=96,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),

    trans2=dict(
        aug_type="scan",
        crop_size=96,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        num_strong_augs=4,
        cutout_kwargs=dict(n_holes=1,
                           length=32,
                           random=True)
    ),
)

solver = dict(
    type="adam",
    base_lr=0.0001,
    bias_lr_factor=1,
    weight_decay=0,
    weight_decay_bias=0,
    target_sub_batch_size=target_sub_batch_size,
    batch_size=batch_size,
    train_sub_batch_size=train_sub_batch_size,
    num_repeat=num_repeat,
)

results = dict(
    output_dir="./results/sediments/{}".format(model_name),
)
