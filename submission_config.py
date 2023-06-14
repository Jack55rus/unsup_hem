from pathlib import Path


class Params:
    checkpoint_path = Path('/home/markin/Markin/prj/innopolis/data/all_ckpts/automask_2/weights_800.pth')
    submission_img_dir = Path('/home/markin/Markin/prj/innopolis/data/submission_slices')  # submission_slices
    submission_pred_dir = Path('/home/markin/Markin/prj/innopolis/data/all_submission_preds')
    submission_vis_dir = Path('/home/markin/Markin/prj/innopolis/data/all_submission_vis')
    unsup_emb = 32
    device = 'cuda'
    csv_path = Path('/home/markin/Markin/prj/innopolis/data/csvs')
    csv_name = ''
    num_files_to_infer = -1  # -1 for all
    threshold = 0.5
