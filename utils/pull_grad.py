import os
import shutil

def copy_matching_images(src_folder, grad_folder, dest_folder):
    # 获取train文件夹中的所有文件名
    train_images = set(os.listdir(src_folder))

    # 遍历grad文件夹中的所有文件
    for root, dirs, files in os.walk(grad_folder):
        for file in files:
            # 如果文件在train中存在，将其复制到grad_fold文件夹下
            if file in train_images:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_folder, file)
                shutil.copy2(src_path, dest_path)

# 创建grad_fold文件夹
src_folder = "../../../autodl-tmp/MaskAll_fold/fold_1"
grad_folder = "../../../autodl-tmp/gradAll"

grad_fold_train = '../../../autodl-tmp/grad_fold_1/train'
os.makedirs(grad_fold_train, exist_ok=True)
grad_fold_val = '../../../autodl-tmp/grad_fold_1/val'
os.makedirs(grad_fold_val, exist_ok=True)


# 调用函数复制匹配的图片
copy_matching_images(os.path.join(src_folder, 'train'), grad_folder, grad_fold_train)
copy_matching_images(os.path.join(src_folder, 'val'), grad_folder, grad_fold_val)

print("复制完成！")
