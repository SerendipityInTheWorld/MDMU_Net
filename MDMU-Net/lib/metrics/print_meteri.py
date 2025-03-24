import nibabel as nib
from medpy.metric.binary import dc, jc,hd95,sensitivity,specificity
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

class meteri3D:
    def __init__(self,class_num) -> None:
        self.class_num = class_num

    def load_volume(self,file_path):
        """使用nibabel加载nii.gz文件为numpy数组"""
        volume = nib.load(file_path)
        return volume.get_fdata()

    def preprocess_data(self,volume, target_class):
        """
        将数据预处理为针对特定类别的二值数据
        """
        return (volume == target_class).astype(int)
    def compute(self,pred_volume, label_volume):
            # 计算Dice系数
            dice_score = dc(pred_volume, label_volume)
            # 计算IoU（使用Jaccard Index）
            iou_score = jc(pred_volume, label_volume)
            # 计算hd95
            if pred_volume.sum() == 0 and label_volume.sum() >0 :
                hd_95 = 512
            else:
                hd_95 = hd95(pred_volume,label_volume)
            # 计算敏感性
            sens = sensitivity(pred_volume,label_volume)
            # 计算特异性
            spec = specificity(pred_volume,label_volume)


            return [dice_score, iou_score,sens,spec,hd_95]

    def compute_metrics_per_class(self,pred_path, label_path):
        """
        计算预测体积和标签体积中特定类别的Dice和IoU
        """
        value_all = []
        dice = []
        iou = []
        sens = []
        spec = []
        hd_95 = []
        # 分别计算类别1和2的Dice和IoU
        for class_id in range(1,self.class_num):
            
            # 加载并预处理数据
            pred_volume = self.preprocess_data(self.load_volume(pred_path), class_id)
            label_volume = self.preprocess_data(self.load_volume(label_path), class_id)
            if np.sum(pred_volume) == 0 and np.sum(label_volume) == 0:
                return np.nan,np.nan,np.nan,np.nan,np.nan
            
            result = self.compute(pred_volume,label_volume)
            value_all.append(result)
        for i in value_all:
            dice.append(i[0])
            iou.append(i[1])
            sens.append(i[2])
            spec.append(i[3])
            hd_95.append(i[4])
        
        return dice,iou,sens,spec,hd_95

        

if __name__ == '__main__':

    # 定义路径,预测结果路径,标签路径,结果保存路径
    pre_result = r'Aeresult3/NIH/BAACA_1_2/seg'
    label_path = r'/root/autodl-tmp/Pancreatic2024/imagesTs'
    save_dir = r'Aeresult3/NIH/BAACA_1_2/seg2'
    class_num = 3 # 类别数量

    my_meteri = meteri3D(class_num=class_num)

    saveCaseName = []
    Dice = []
    Iou = []
    Sens = []
    Spec = []
    HD95_list = []

    labelFileName = sorted(os.listdir(label_path))

    for name in tqdm(labelFileName):

        if not name.startswith('.') and name.endswith('nii.gz'):
            print(name)
            prediction_file = os.path.join(pre_result,name)
            label_file = os.path.join(label_path,name)

            saveCaseName.append(name)
            # 分别计算类别1和2的Dice和IoU
            dice, iou, sens,spec,hd_95 = my_meteri.compute_metrics_per_class(prediction_file, label_file)
            
            meanDice = np.nanmean(dice)
            dice.append(meanDice)
            Dice.append(dice)

            meanIou = np.nanmean(iou)
            iou.append(meanIou)
            Iou.append(iou)

            meanSens = np.nanmean(sens)
            sens.append(meanSens)
            Sens.append(sens)

            meanSpec = np.nanmean(spec)
            spec.append(meanSpec)
            Spec.append(spec)

            meanHD95 = np.nanmean(hd_95)
            hd_95.append(meanHD95)
            HD95_list.append(hd_95)

    all_data = []
    for d,i,sens,spec,h in zip(Dice,Iou,Sens,Spec,HD95_list):
        all_data_splic = []
        for j in d:
            all_data_splic.append(j)
        for j in i:
            all_data_splic.append(j)
        for j in sens:
            all_data_splic.append(j)
        for j in spec:
            all_data_splic.append(j)
        for j in h:
            all_data_splic.append(j)
        all_data.append(all_data_splic)

    all_columns = ['Dice_1','Dice_2','meanDice','Iou_1','Iou_2','meanIou',
                    'Sens_1','Sens_2','meanSens','Spec_1','Spec_2','meanSpec',
                    'HD95_1','HD95_2','meanHD95']

    df = pd.DataFrame(data=all_data,columns=all_columns,index=saveCaseName)
    df.to_csv(os.path.join(save_dir,'MMNet_NIH_1.csv'))

                
