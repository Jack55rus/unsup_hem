from pathlib import Path

PROJECT_PATH = Path(__file__).parents[0]
DATA_PATH = PROJECT_PATH / 'data'

class Params:
    name = 'pseudo_2'
    epochs = 1
    batch_size = 4
    lr = 0.0005
    weight_decay = 0.0005
    scheduler_patience = 2
    scheduler_factor = 0.2
    ce_margin = 0.2
    unsup_emb = 32
    auto_mask_dir = DATA_PATH / 'pseudo' # Path('/home/markin/Markin/prj/innopolis/data/pathology/pseudo_2')
    img_root_dir = DATA_PATH / 'slices' # Path('/home/markin/Markin/prj/innopolis/data/slices')
    device = 'cuda'
    ckpt_save_path = DATA_PATH / f'all_ckpts/{name}' #  Path(f'/home/markin/Markin/prj/innopolis/data/all_ckpts/{name}')
    seed = 3407
    loss_print_interval = 20
    ckpt_save_interval = 100
    pred_save_dir = DATA_PATH / f'all_predictions/{name}' # Path(f'/home/markin/Markin/prj/innopolis/data/all_predictions/{name}')
    ckpt_load_path = None
    comment = 'cleaned decision'
    log_dir = DATA_PATH / f'plots/{name}'  # Path(f'/home/markin/Markin/prj/innopolis/data/plots/{name}')



