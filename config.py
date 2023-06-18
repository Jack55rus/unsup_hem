from pathlib import Path
class Params:
    name = 'automask_1'
    epochs = 10
    batch_size = 2
    lr = 0.0005  # 0.0005
    weight_decay = 0.0005
    scheduler_patience = 2
    scheduler_factor = 0.2
    ce_margin = 0.2
    unsup_emb = 32
    auto_mask_dir = Path('/home/markin/Markin/prj/innopolis/data/pathology/masks')
    img_root_dir = Path('/home/markin/Markin/prj/innopolis/data/slices')
    device = 'cuda'
    ckpt_save_path = Path(f'/home/markin/Markin/prj/innopolis/data/all_ckpts/{name}')
    seed = 504
    loss_print_interval = 20
    ckpt_save_interval = 100
    pred_save_dir = Path(f'/home/markin/Markin/prj/innopolis/data/all_predictions/{name}')
    ckpt_load_path = None # Path('/home/markin/Markin/prj/innopolis/data/all_ckpts/automask_2_64/weights_1000.pth')
    comment = 'new train with initial masks'
    log_dir = Path(f'/home/markin/Markin/prj/innopolis/data/plots/{name}')


