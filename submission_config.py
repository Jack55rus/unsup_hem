from pathlib import Path


class Params:
    checkpoint_path = Path('/home/markin/Markin/prj/innopolis/data/all_ckpts/automask_2/weights_1000.pth')
    checkpoint_path_add_1 = None # Path('/home/markin/Markin/prj/innopolis/data/all_ckpts/automask_2/weights_1000.pth')  # best
    checkpoint_path_add_2 = None # Path('/home/markin/Markin/prj/innopolis/data/all_ckpts/automask_1/weights_1000.pth')
    submission_img_dir = Path('/home/markin/Markin/prj/innopolis/data/submission_slices_2')  # submission_slices
    submission_pred_dir = Path('/home/markin/Markin/prj/innopolis/data/all_submission_preds')  #
    submission_vis_dir = Path('/home/markin/Markin/prj/innopolis/data/all_submission_vis')
    unsup_emb = 32
    device = 'cuda'
    csv_path = Path('/home/markin/Markin/prj/innopolis/data/csvs')
    csv_name = ''
    num_files_to_infer = 1000  # -1 for all
    threshold = 0.5  # 0.7
    threshold_add_1 = 0.8
    threshold_add_2 = 0.8
    smallest_size = 25  # 400
    ensemble_method = 'or'
    erosion_window = 15
