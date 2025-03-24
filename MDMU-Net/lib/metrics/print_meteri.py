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
        volume = nib.load(file_path)
        return volume.get_fdata()

    def preprocess_data(self,volume, target_class):
        return (volume == target_class).astype(int)
    def compute(self,pred_volume, label_volume):

            dice_score = dc(pred_volume, label_volume)

            iou_score = jc(pred_volume, label_volume)

            if pred_volume.sum() == 0 and label_volume.sum() >0 :
                hd_95 = 512
            else:
                hd_95 = hd95(pred_volume,label_volume)

            sens = sensitivity(pred_volume,label_volume)

            spec = specificity(pred_volume,label_volume)


            return [dice_score, iou_score,sens,spec,hd_95]

    def compute_metrics_per_class(self,pred_path, label_path):

        value_all = []
        dice = []
        iou = []
        sens = []
        spec = []
        hd_95 = []

        for class_id in range(1,self.class_num):
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

    pre_result = r''
    label_path = r''
    save_dir = r''
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
    df.to_csv(os.path.join(save_dir,'name.csv'))

                
