import os
import requests
import yaml
import glob
from tqdm import tqdm
import argparse

def download_weights(url,file_save_name):
    """
    Download weights for any model.

    :param url: Dwonload url for the weight file.
    :param file_save_name:  String name to save the file on to the disk.
    """
    # Make checkpoint directory if not present
    data_dir = 'checkpoints'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, file_save_name)

    # Download the file if not present
    if not os.path.exists(file_path):
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            progress_bar = tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {file_save_name}",
                colour="green"
            )

            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))

            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                print("⚠️ Downloaded file size mismatch!")

def parse_meta_file():
    """
    Function to parse all the model files inside 'mmdetection/configs'
    and return the download URLs for all the available models.

    Returns:
        weight_list: List containing URLs for all the downloadble models.
    """
    root_meta_file_path='mmdetection/configs'
    all_meta_file_paths=glob.glob(os.path.join(root_meta_file_path,"*","metafile.yml"),recursive=True)
    weights_list = []

    # j = 0
    for meta_file_path in all_meta_file_paths:
        with open(meta_file_path,'r') as f:
            yaml_file = yaml.safe_load(f)
            yaml_file_models = None
            # Sometimes the yaml file loads as dictionary and sometimes a list
            if isinstance(yaml_file,dict):
                yaml_file_models = yaml_file["Models"]
            else:
                yaml_file_models = yaml_file
            for i in range(len(yaml_file_models)):
                try:
                    # print(f"yaml_file {j}: {yaml_file_models[i]['Weights']}\n")
                    weights_list.append(yaml_file_models[i]['Weights'])
                except:
                    metrics = yaml_file_models[i]['Results'][0].get('Metrics',None)
                    if(metrics):
                        weights = yaml_file_models[i]['Results'][0]['Metrics'].get('Weights',None)
                        if(weights):
                            weights_list.append(weights)
                    else:
                        # weights_list.append(f"no url found for model:{yaml_file_models[i]['Name']}")
                        weights_list.append(f"no url found for model: {yaml_file_models[i].get('Name', 'Unknown')}")

            # j=j+1   
    

    return weights_list


def format_checkpoint_file_name(folder_list, checkpoint_file_name):
    """
    Format checkpoint file names if the parent folder matches special folders.

    Args:
        folder_list (list): List of folder names requiring formatting.
        checkpoint_file_name (str): Path to the checkpoint config file.

    Returns:
        str: Formatted checkpoint file name.
    """
    # Normalize path separators to handle both \ and /
    parts = checkpoint_file_name.replace("\\", "/").split("/")

    # Get the folder name before the file (e.g., 'dino' or 'double_heads')
    folder_name = parts[-2]

    if folder_name in folder_list:
            filename = parts[-1]
            
            # Replace only the first 2 underscores with hyphens
            underscore_count = 0
            new_filename = ""
            for ch in filename:
                if ch == "_" and underscore_count < 2:
                    new_filename += "-"
                    underscore_count += 1
                else:
                    new_filename += ch    
            parts[-1] = new_filename

    # Rebuild with backslashes
    return "\\".join(parts)


def get_model(weights_name):
    """
    Either downloads a model or loads one from local path if already
    downloaded using the weight file name ('weights_name') provided.

    :param weights_name: Name of the weight file. Most like in the format:
    **'retinanet_ghm_r50_fpn_1x_coco'**. See **'weights.txt'** to know weight file name formats and downloadable URL formats.

    Returns:
        model: The loaded detection model.
    """
    # Get the list containing all the weight file download URLs.
    weights_list = parse_meta_file()

    download_url = None
    for weights in weights_list:
        if weights_name in weights:
            print(f'Found weights: {weights}\n')
            download_url = weights
            break
        

    assert download_url != None, f"{weights_name} weight file not found!!!"
    
    # Download the checkpoint file.(will only donwload if the checkpoint file is not present locally)
    download_weights(url=download_url,file_save_name=download_url.split('/')[-1])

    # checkpoint_file = os.path.join('checkpoint',download_url.split('/')[-1])

    # # Build the model using the configuration file.
    # config_file = os.path.join(
    #     'mmdetection/configs',
    #     download_url.split('/')[-3],
    #     download_url.split('/')[-2]+
    #     '.py')
    
    # special_file_name_containing_folders = ["dino", "double_heads"]

    # config_file = format_checkpoint_file_name(special_file_name_containing_folders,config_file)

    # model = init_detector(config=config_file,checkpoint=checkpoint_file,cfg_options=dict(
    #     rcnn=dict(
    #         score_thr=0.05,   # detection threshold
    #         nms=dict(type='nms', iou_threshold=0.5),
    #         max_per_img=100
    #     )
    # ))

    # return model

def write_weights_txt_file():
    """
    Write all the model URLs to 'weights.txt' to have complete list and choose one of them.add()

    EXECUTE 'utils.py' if 'weights.txt' not already present.
    'python utils.py' command will generate the latest 'weights.txt'
    file according to the cloned mmdetection repository.
    """

    # Get the list containing all the weight file download URLs
    weight_list = parse_meta_file()
    with open('weights.txt','w') as f:
        for weights in weight_list:
            f.writelines(f"{weights}\n")
        f.close()


if __name__ == '__main__':
    write_weights_txt_file()

    # Construct the argument parser.
    parser = argparse.ArgumentParser(description="tool for object recognition in images")

    parser.add_argument(
        '-w','--weights',default='faster_rcnn_r50_fpn_1x_coco',
        help='weight file name'
    )

    args = vars(parser.parse_args()) # will take the input from the commandline and construct a dictionary out of it.
    get_model(args['weights'])
    weights_list = parse_meta_file()
    for i in range(0,3):
        print(weights_list[i])