from pathlib import Path
class Params:
    name = 'automask_2'
    epochs = 10
    batch_size = 2
    lr = 0.0005
    weight_decay = 0.0005
    scheduler_patience = 2
    scheduler_factor = 0.2
    ce_margin = 0.2
    unsup_emb = 32
    auto_mask_dir = Path('/home/markin/Markin/prj/innopolis/data/pathology/pseudo_2')
    img_root_dir = Path('/home/markin/Markin/prj/innopolis/data/slices')
    device = 'cuda'
    ckpt_save_path = Path(f'/home/markin/Markin/prj/innopolis/data/all_ckpts/{name}')
    seed = 201
    loss_print_interval = 100
    pred_save_dir = Path(f'/home/markin/Markin/prj/innopolis/data/all_predictions/{name}')
    ckpt_load_path = None  # Path('/home/markin/Markin/prj/innopolis/data/all_ckpts/more_augs/weights_600.pth')
    comment = 'generated new auto masks for stage 1'
    log_dir = Path(f'/home/markin/Markin/prj/innopolis/data/plots/{name}')


