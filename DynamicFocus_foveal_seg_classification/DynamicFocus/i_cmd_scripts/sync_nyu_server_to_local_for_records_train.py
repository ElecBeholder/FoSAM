import subprocess

import preset


def copy_remote_folder_to_local(remote_host, remote_user, remote_password, remote_path, local_path):
    try:
        # 构建 pscp 命令
        command = [
            "pscp", "-r",  # -r 表示递归拷贝整个文件夹
            "-pw", remote_password,  # 远程服务器密码
            f"{remote_user}@{remote_host}:{remote_path}/*",  # 远程路径
            local_path  # 本地路径（包含文件夹本身）
        ]

        # 执行命令
        result = subprocess.run(command, check=True, shell=True)

        if result.returncode == 0:
            print(f'fm {remote_path}')
            print(f'to {local_path}')
            print("文件夹拷贝成功")
    except subprocess.CalledProcessError as e:
        print(f"执行 pscp 命令失败: {e}")
    except Exception as e:
        print(f"出现错误: {e}")


# copy_remote_folder_to_local('172.24.113.253', 'hongyiz', 'hongyiz', preset.dpath_training_records_server, preset.dpath_training_records_local)
copy_remote_folder_to_local('172.24.113.252', 'hongyiz', 'hongyiz001', preset.dpath_training_records_server_4090_or_H100, preset.dpath_training_records_harry_local)

r"""
pscp -r D:\b_data_train hongyiz@172.24.113.252:/home/hongyiz/DriverD

pscp -r D:\b_data_train\data_a_raw\coco2017 hongyiz@172.24.113.252:/home/hongyiz/DriverD/b_data_train/data_a_raw
pscp -r D:\b_data_train\data_a_raw\lvis_v1_train hongyiz@172.24.113.252:/home/hongyiz/DriverD/b_data_train/data_a_raw
pscp -r D:\b_data_train\data_a_raw\lvis_v1_val hongyiz@172.24.113.252:/home/hongyiz/DriverD/b_data_train/data_a_raw


pscp -r D:\b_data_train\data_a_raw\gtFine_trainvaltest hongyiz@172.24.113.252:/home/hongyiz/DriverD/b_data_train/data_a_raw
pscp -r D:\b_data_train\data_a_raw\leftImg8bit_trainvaltest hongyiz@172.24.113.252:/home/hongyiz/DriverD/b_data_train/data_a_raw

"""
