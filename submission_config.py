from config import DATA_PATH


class Params:
    checkpoint_path = DATA_PATH / 'weights.pth' # Path('/home/markin/Markin/prj/innopolis/data/all_ckpts/automask_2/weights_1000.pth')
    submission_img_dir = DATA_PATH / 'submission_slices_2'  # Path('/home/markin/Markin/prj/innopolis/data/submission_slices_2')
    submission_pred_dir = DATA_PATH / 'all_submission_preds' # Path('/home/markin/Markin/prj/innopolis/data/all_submission_preds')
    submission_vis_dir = DATA_PATH / 'all_submission_vis' # Path('/home/markin/Markin/prj/innopolis/data/all_submission_vis')
    unsup_emb = 32
    device = 'cuda'
    csv_path = DATA_PATH / 'csvs'  # Path('/home/markin/Markin/prj/innopolis/data/csvs')
    csv_name = ''
    num_files_to_infer = -1  # -1 for all
    threshold = 0.5
    smallest_size = 25
    erosion_window = 20
