from pathlib import Path

PROJECT_PATH = Path(__file__).parents[1]
DATA_Path = PROJECT_PATH / 'data'

class Params:
    name = 'smaller_net'
    epochs = 10
    batch_size = 4
    lr = 0.0005  # 0.0005
    weight_decay = 0.0005
    scheduler_patience = 2
    scheduler_factor = 0.2
    ce_margin = 0.2
    unsup_emb = 32
    auto_mask_dir = DATA_Path / 'pathology/pseudo_2' # Path('/home/markin/Markin/prj/innopolis/data/pathology/pseudo_2')
    img_root_dir = DATA_Path / 'slices' # Path('/home/markin/Markin/prj/innopolis/data/slices')
    device = 'cuda'
    ckpt_save_path = Path(f'/home/markin/Markin/prj/innopolis/data/all_ckpts/{name}')
    seed = 3407
    loss_print_interval = 20
    ckpt_save_interval = 100
    pred_save_dir = Path(f'/home/markin/Markin/prj/innopolis/data/all_predictions/{name}')
    ckpt_load_path = None # Path('/home/markin/Markin/prj/innopolis/data/all_ckpts/larger_net/weights_600.pth')
    comment = 'reduced network'
    log_dir = Path(f'/home/markin/Markin/prj/innopolis/data/plots/{name}')



