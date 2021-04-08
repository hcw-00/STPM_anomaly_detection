import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
import shutil
import time
from resnet_modified import resnet18
from PIL import Image
from sklearn.metrics import roc_auc_score


mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def data_transforms(load_size=256, input_size=256):
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size)),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
        
    return data_transforms

def data_transforms_inv():
    data_transforms_inv = transforms.Compose([transforms.Normalize(mean=list(-np.divide(mean_train, std_train)), std=list(np.divide(1, std_train)))])
    return data_transforms_inv

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)


def cal_loss(fs_list, ft_list, criterion):
    tot_loss = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = torch.divide(fs, torch.norm(fs, p=2, dim=1, keepdim=True))
        ft_norm = torch.divide(ft, torch.norm(ft, p=2, dim=1, keepdim=True))
        f_loss = (0.5/(w*h))*criterion(fs_norm, ft_norm)
        tot_loss += f_loss
    return tot_loss

def cal_anomaly_map(fs_list, ft_list, out_size=256):
    pdist = torch.nn.PairwiseDistance(p=2, keepdim=True)
    anomaly_map = torch.ones([ft_list[0].shape[0], 1, out_size, out_size]).to(device)
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = torch.divide(fs, torch.norm(fs, p=2, dim=1, keepdim=True))
        ft_norm = torch.divide(ft, torch.norm(ft, p=2, dim=1, keepdim=True))
        a_map = 0.5*pdist(fs_norm, ft_norm)**2
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear')
        anomaly_map *= a_map
    return anomaly_map

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--dataset_path', default=r'D:\Dataset\mvtec_anomaly_detection\bottle')
    parser.add_argument('--num_epoch', default=100)
    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--project_path', default='D:/Project_Train_Results/mvtec_anomaly_detection/bottle')
    parser.add_argument('--save_weight', default=False)
    parser.add_argument('--save_src_code', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

    ################################################
    ###             Set Parameters               ###
    ################################################
    
    args = get_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epoch
    lr = args.lr
    batch_size = args.batch_size
    save_weight = args.save_weight
    load_size = args.load_size
    input_size = args.input_size
    save_src_code = args.save_src_code
    project_path = args.project_path
    logs_path = os.path.join(project_path, 'logs')
    weight_save_path = os.path.join(project_path, 'saved')
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(weight_save_path, exist_ok=True)
    if save_src_code:
        source_code_save_path = os.path.join(project_path, 'src')
        os.makedirs(source_code_save_path, exist_ok=True)
        copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README']) # copy source code

    ################################################
    ###             Define Dataset               ###
    ################################################

    data_transform = data_transforms(input_size=input_size)
    # data_transforms_inv = data_transforms_inv()
    image_datasets = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'), transform=data_transform)
    dataloaders = DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataset_sizes = {'train': len(image_datasets)}

    ################################################
    ###             Define Network               ###
    ################################################
    model_t = resnet18(pretrained=True, num_classes=1).to(device)
    model_s = resnet18(pretrained=False, num_classes=1).to(device)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model_s.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    
    ################################################
    ###               Start Train                ###
    ################################################

    start_time = time.time()
    global_step = 0
    print('Dataset size : Train set - {}'.format(dataset_sizes['train']))
    for epoch in range(num_epochs):
        print('-'*20)
        print('Time consumed : {}s'.format(time.time()-start_time))
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*20)
        model_t.eval()
        model_s.train()
        for idx, (batch, labels) in enumerate(dataloaders): # batch loop
            global_step += 1
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                _, f2t, f3t, f4t, f5t = model_t(batch)
                _, f2s, f3s, f4s, f5s = model_s(batch)
                loss = cal_loss([f2s,f3s,f4s], [f2t,f3t,f4t], criterion)
                loss.backward()
                optimizer.step()
            if idx%2 == 0:
                print('Epoch : {} | Loss : {:.4f}'.format(epoch, float(loss.data)))

    print('Total time consumed : {}'.format(time.time() - start_time))
    print('Train end.')

    ################################################
    ###               Start Test                 ###
    ################################################

    print('Test phase start')
    model_s.eval() # check
    model_t.eval()
    # anomaly_maps = []
    # ground_truths = []
    test_path = os.path.join(dataset_path, 'test')
    gt_path = os.path.join(dataset_path, 'ground_truth')
    test_imgs = glob.glob(test_path + '/[!good]*/*.png', recursive=True)
    gt_imgs = glob.glob(gt_path + '/[!good]*/*.png', recursive=True)
    test_transform = data_transforms()
    auc_score_list = []
    start_time = time.time()
    for i in range(len(test_imgs)):
        test_img_path = test_imgs[i]
        gt_img_path = gt_imgs[i]
        assert os.path.split(test_img_path)[1].split('.')[0] == os.path.split(gt_img_path)[1].split('_')[0], "Something wrong with test and ground truth pair!"
        test_img_o = cv2.imread(test_img_path)
        test_img = Image.fromarray(test_img_o)
        test_img = test_transform(test_img)
        test_img = torch.unsqueeze(test_img, 0).to(device)
        with torch.set_grad_enabled(False):
            _, f2t, f3t, f4t, f5t = model_t(test_img)
            _, f2s, f3s, f4s, f5s = model_s(test_img)
        anomaly_map = cal_anomaly_map([f2s,f3s,f4s], [f2t,f3t,f4t], out_size=input_size)
        anomaly_map = anomaly_map[0,0,:,:].to('cpu').detach().numpy().ravel()
        gt_img = cv2.imread(gt_img_path,0)
        gt_img = cv2.resize(gt_img, (input_size, input_size)).ravel()//255
        
        auc_score_list.append(roc_auc_score(gt_img, anomaly_map))

    print('Total test time consumed : {}'.format(time.time() - start_time))
    print("Total auc score is :")
    print(np.mean(auc_score_list))

