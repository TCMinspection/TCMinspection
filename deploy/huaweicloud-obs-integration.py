#!/usr/bin/env python3
# 华为云OBS对象存储集成
# 用于模型文件和用户数据的云存储

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import threading
import time

try:
    from obs import ObsClient
    from obs import LoadConfiguration, GetObjectRequest, PutObjectRequest
    from obs import ListObjectsRequest, DeleteObjectRequest
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False
    print("华为云OBS SDK未安装，请运行: pip install esdk-obs-python")

class HuaweiCloudOBSManager:
    """华为云OBS对象存储管理器"""

    def __init__(self, config_file: str = None):
        """
        初始化OBS管理器

        Args:
            config_file: 配置文件路径
        """
        self.logger = self._setup_logger()
        self.obs_client = None
        self.bucket_name = None

        if OBS_AVAILABLE:
            self._initialize_client(config_file)
        else:
            self.logger.error("华为云OBS SDK不可用")

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('huaweicloud_obs')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_client(self, config_file: str = None):
        """初始化OBS客户端"""
        try:
            # 从环境变量获取配置
            access_key = os.getenv('HUAWEICLOUD_SDK_AK') or os.getenv('OBS_ACCESS_KEY')
            secret_key = os.getenv('HUAWEICLOUD_SDK_SK') or os.getenv('OBS_SECRET_KEY')
            endpoint = os.getenv('OBS_ENDPOINT', 'https://obs.cn-north-4.myhuaweicloud.com')
            self.bucket_name = os.getenv('OBS_BUCKET_NAME', 'tcm-inspection-bucket')

            if not all([access_key, secret_key]):
                raise ValueError("华为云认证信息不完整")

            # 创建OBS客户端
            config = LoadConfiguration()
            config.connect_timeout = 30
            config.socket_timeout = 60
            config.chunk_size = 65536

            self.obs_client = ObsClient(
                access_key_id=access_key,
                secret_access_key=secret_key,
                server=endpoint,
                security_provider_policy=config
            )

            # 确保桶存在
            self._ensure_bucket_exists()

            self.logger.info(f"华为云OBS客户端初始化成功，桶名: {self.bucket_name}")

        except Exception as e:
            self.logger.error(f"OBS客户端初始化失败: {str(e)}")
            self.obs_client = None

    def _ensure_bucket_exists(self):
        """确保存储桶存在"""
        try:
            # 尝试创建桶
            resp = self.obs_client.createBucket(self.bucket_name)
            if resp.status < 300:
                self.logger.info(f"创建存储桶 {self.bucket_name} 成功")
            else:
                # 桶可能已存在，这是正常的
                self.logger.info(f"存储桶 {self.bucket_name} 已存在")
        except Exception as e:
            self.logger.error(f"创建存储桶失败: {str(e)}")

    def upload_file(self, local_file_path: str, object_key: str = None) -> bool:
        """
        上传文件到OBS

        Args:
            local_file_path: 本地文件路径
            object_key: OBS对象键，如果为None则使用文件名

        Returns:
            bool: 上传是否成功
        """
        if not self.obs_client:
            self.logger.error("OBS客户端未初始化")
            return False

        if not os.path.exists(local_file_path):
            self.logger.error(f"本地文件不存在: {local_file_path}")
            return False

        if object_key is None:
            object_key = os.path.basename(local_file_path)

        try:
            with open(local_file_path, 'rb') as file_data:
                resp = self.obs_client.putObject(
                    bucketName=self.bucket_name,
                    objectKey=object_key,
                    content=file_data.read()
                )

            if resp.status < 300:
                self.logger.info(f"文件上传成功: {object_key}")
                return True
            else:
                self.logger.error(f"文件上传失败: {resp.errorMessage}")
                return False

        except Exception as e:
            self.logger.error(f"文件上传异常: {str(e)}")
            return False

    def download_file(self, object_key: str, local_file_path: str) -> bool:
        """
        从OBS下载文件

        Args:
            object_key: OBS对象键
            local_file_path: 本地保存路径

        Returns:
            bool: 下载是否成功
        """
        if not self.obs_client:
            self.logger.error("OBS客户端未初始化")
            return False

        try:
            resp = self.obs_client.getObject(
                bucketName=self.bucket_name,
                objectKey=object_key
            )

            if resp.status < 300:
                # 确保目录存在
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                with open(local_file_path, 'wb') as file_data:
                    for chunk in resp.body.response:
                        file_data.write(chunk)

                self.logger.info(f"文件下载成功: {object_key} -> {local_file_path}")
                return True
            else:
                self.logger.error(f"文件下载失败: {resp.errorMessage}")
                return False

        except Exception as e:
            self.logger.error(f"文件下载异常: {str(e)}")
            return False

    def list_files(self, prefix: str = "", max_keys: int = 100) -> list:
        """
        列出OBS中的文件

        Args:
            prefix: 对象键前缀
            max_keys: 最大返回数量

        Returns:
            list: 文件列表
        """
        if not self.obs_client:
            self.logger.error("OBS客户端未初始化")
            return []

        try:
            resp = self.obs_client.listObjects(
                bucketName=self.bucket_name,
                prefix=prefix,
                max_keys=max_keys
            )

            if resp.status < 300:
                files = []
                for content in resp.body.contents:
                    files.append({
                        'key': content.key,
                        'size': content.size,
                        'last_modified': content.lastModified,
                        'etag': content.etag
                    })
                return files
            else:
                self.logger.error(f"列出文件失败: {resp.errorMessage}")
                return []

        except Exception as e:
            self.logger.error(f"列出文件异常: {str(e)}")
            return []

    def delete_file(self, object_key: str) -> bool:
        """
        删除OBS中的文件

        Args:
            object_key: OBS对象键

        Returns:
            bool: 删除是否成功
        """
        if not self.obs_client:
            self.logger.error("OBS客户端未初始化")
            return False

        try:
            resp = self.obs_client.deleteObject(
                bucketName=self.bucket_name,
                objectKey=object_key
            )

            if resp.status < 300:
                self.logger.info(f"文件删除成功: {object_key}")
                return True
            else:
                self.logger.error(f"文件删除失败: {resp.errorMessage}")
                return False

        except Exception as e:
            self.logger.error(f"文件删除异常: {str(e)}")
            return False

    def upload_directory(self, local_dir: str, obs_prefix: str = "") -> bool:
        """
        上传整个目录到OBS

        Args:
            local_dir: 本地目录路径
            obs_prefix: OBS前缀

        Returns:
            bool: 上传是否成功
        """
        if not os.path.exists(local_dir):
            self.logger.error(f"本地目录不存在: {local_dir}")
            return False

        success_count = 0
        total_count = 0

        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_dir)
                object_key = f"{obs_prefix}/{relative_path}".replace("\\", "/")

                total_count += 1
                if self.upload_file(local_file_path, object_key):
                    success_count += 1

        self.logger.info(f"目录上传完成: {success_count}/{total_count} 个文件成功")
        return success_count == total_count

    def backup_models(self, models_dir: str = "/home/tcmuser/tcm-inspection/models") -> bool:
        """
        备份模型文件到OBS

        Args:
            models_dir: 模型文件目录

        Returns:
            bool: 备份是否成功
        """
        if not os.path.exists(models_dir):
            self.logger.warning(f"模型目录不存在: {models_dir}")
            return True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_prefix = f"backups/models/{timestamp}"

        return self.upload_directory(models_dir, backup_prefix)

    def backup_reports(self, reports_dir: str = "/home/tcmuser/tcm-inspection/reports") -> bool:
        """
        备份报告文件到OBS

        Args:
            reports_dir: 报告文件目录

        Returns:
            bool: 备份是否成功
        """
        if not os.path.exists(reports_dir):
            self.logger.warning(f"报告目录不存在: {reports_dir}")
            return True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_prefix = f"backups/reports/{timestamp}"

        return self.upload_directory(reports_dir, backup_prefix)

    def get_public_url(self, object_key: str, expires_in: int = 3600) -> Optional[str]:
        """
        获取对象的公开访问URL

        Args:
            object_key: OBS对象键
            expires_in: 过期时间（秒）

        Returns:
            str: 公开URL
        """
        if not self.obs_client:
            self.logger.error("OBS客户端未初始化")
            return None

        try:
            resp = self.obs_client.createSignedUrl(
                'GET',
                self.bucket_name,
                object_key,
                expires_in
            )

            if resp.status < 300:
                return resp.body.signedUrl
            else:
                self.logger.error(f"生成签名URL失败: {resp.errorMessage}")
                return None

        except Exception as e:
            self.logger.error(f"生成签名URL异常: {str(e)}")
            return None

    def cleanup_old_backups(self, days_to_keep: int = 30) -> bool:
        """
        清理旧的备份文件

        Args:
            days_to_keep: 保留天数

        Returns:
            bool: 清理是否成功
        """
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)

        # 清理模型备份
        model_files = self.list_files("backups/models/", 1000)
        deleted_count = 0

        for file_info in model_files:
            try:
                last_modified = datetime.strptime(
                    file_info['last_modified'],
                    '%Y-%m-%dT%H:%M:%S.%fZ'
                ).timestamp()

                if last_modified < cutoff_date:
                    if self.delete_file(file_info['key']):
                        deleted_count += 1
            except:
                continue

        self.logger.info(f"清理完成，删除了 {deleted_count} 个旧备份文件")
        return True


def main():
    """测试OBS管理器功能"""
    manager = HuaweiCloudOBSManager()

    if not manager.obs_client:
        print("OBS管理器初始化失败")
        return

    # 测试上传文件
    test_file = "/tmp/test.txt"
    with open(test_file, 'w') as f:
        f.write("这是一个测试文件")

    if manager.upload_file(test_file, "test/test.txt"):
        print("文件上传成功")

        # 测试下载文件
        download_path = "/tmp/downloaded_test.txt"
        if manager.download_file("test/test.txt", download_path):
            print("文件下载成功")

        # 测试删除文件
        if manager.delete_file("test/test.txt"):
            print("文件删除成功")

    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)


if __name__ == "__main__":
    main()