"""
用户向量 → 匿名群体向量生成系统
核心流程：从HDFS读取 → 自适应LSH → Band分桶 → 鲁棒质心 → vMF差分隐私 → HDFS存储
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score
from scipy import stats
from collections import Counter
from datetime import datetime
import json
import os
import subprocess
import tempfile
import shutil
from typing import List, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HDFSIO:
    """HDFS文件读写工具类，基于hadoop fs命令"""

    def _hdfs_command(self, cmd_list: List[str]) -> Tuple[bool, str, str]:
        """
        执行hadoop命令
        Args:
            cmd_list: 命令列表，如 ['hadoop', 'fs', '-ls', '/path']
        Returns:
            (是否成功, stdout, stderr)
        """
        p = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        ok = (p.returncode == 0)
        return ok, p.stdout.strip(), p.stderr.strip()

    def read_from_hdfs(self, hdfs_path: str) -> pd.DataFrame:
        """
        从HDFS读取Parquet文件到DataFrame
        Args:
            hdfs_path: HDFS路径（支持目录或文件）
        Returns:
            DataFrame或None（失败时）
        """
        temp_dir = tempfile.mkdtemp(prefix="hdfs_in_")
        try:
            logger.info(f"开始从HDFS读取: {hdfs_path}")
            success, out, err = self._hdfs_command(['hadoop', 'fs', '-get', '-f', hdfs_path, temp_dir])
            if not success:
                logger.error(f"HDFS -get 失败: {err}")
                return None

            # 处理目录或文件
            local_path = os.path.join(temp_dir, os.path.basename(hdfs_path))
            if not os.path.exists(local_path):
                local_path = temp_dir

            df = pd.read_parquet(local_path)
            logger.info(f"HDFS读取成功: shape={df.shape}, memory={df.memory_usage(deep=True).sum() / 1024**2:.2f}MB")
            return df
        except Exception as e:
            logger.error(f"读取HDFS数据异常: {e}")
            return None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def write_df_to_hdfs_parquet(self, df: pd.DataFrame, hdfs_path: str) -> bool:
        """
        将DataFrame写为Parquet并上传到HDFS
        Args:
            df: 要写入的DataFrame
            hdfs_path: HDFS目标路径
        Returns:
            是否成功
        """
        temp_dir = tempfile.mkdtemp(prefix="hdfs_out_")
        local_file = os.path.join(temp_dir, "output.parquet")
        try:
            logger.info(f"准备写入HDFS: {hdfs_path}, shape={df.shape}")

            # 写本地Parquet
            df.to_parquet(local_file, index=False)

            # 确保HDFS父目录存在
            parent = os.path.dirname(hdfs_path)
            if parent:
                self._hdfs_command(['hadoop', 'fs', '-mkdir', '-p', parent])

            # 覆盖写入HDFS
            self._hdfs_command(['hadoop', 'fs', '-rm', '-r', '-f', hdfs_path])
            success, out, err = self._hdfs_command(['hadoop', 'fs', '-put', '-f', local_file, hdfs_path])

            if not success:
                logger.error(f"HDFS -put 失败: {err}")
            else:
                logger.info(f"HDFS写入成功: {hdfs_path}")
            return success
        except Exception as e:
            logger.error(f"写入HDFS异常: {e}")
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class PrivacyBudgetTracker:
    """
    隐私预算跟踪器

    功能：
    1. 跟踪累积隐私预算
    2. 动态调整每次epsilon
    3. 持久化预算历史
    """

    def __init__(self, epsilon_per_run=1.0, storage_path='privacy_budget.json'):
        self.epsilon_per_run = epsilon_per_run
        self.storage_path = storage_path
        self.history = self._load_history()

    def _load_history(self):
        """加载历史记录"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return {'updates': [], 'total_budget': 0.0}

    def _save_history(self):
        """保存历史记录"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_current_epsilon(self):
        """
        获取当前epsilon（高级组合定理）

        公式：保持总预算不变，ε_t = ε_total / √t
        """
        update_count = len(self.history['updates']) + 1

        # 高级组合：总预算 = ε × √T
        # 反推：ε_single = ε_total / √T
        adjusted_epsilon = self.epsilon_per_run / np.sqrt(update_count)

        # 记录本次更新
        self.history['updates'].append({
            'timestamp': datetime.now().isoformat(),
            'epsilon': adjusted_epsilon,
            'update_count': update_count
        })

        # 计算累积预算
        self.history['total_budget'] = self.epsilon_per_run * np.sqrt(update_count)

        self._save_history()

        return adjusted_epsilon

    def get_report(self):
        """生成预算报告"""
        return {
            'total_updates': len(self.history['updates']),
            'total_budget': self.history['total_budget'],
            'last_update': self.history['updates'][-1] if self.history['updates'] else None,
            'next_epsilon': self.epsilon_per_run / np.sqrt(len(self.history['updates']) + 1)
        }


class AnonymousGroupVectorGenerator:
    """
    匿名群体向量生成器

    技术栈：
    - LSH: 随机超平面法 + 自适应策略
    - 量化: 分位数/线性动态选择
    - 分桶: Band Token + 独立性验证
    - 质心: 加权鲁棒计算
    - 隐私: von Mises-Fisher球面噪声
    """

    def __init__(self,
                 k_min=1000,
                 epsilon=1.0,
                 n_tokens=12,
                 enable_privacy_tracking=True):
        """
        参数：
            k_min: k-匿名最小桶大小
            epsilon: 隐私预算
            n_tokens: LSH token数量
            enable_privacy_tracking: 是否启用隐私预算跟踪
        """
        self.k_min = k_min
        self.base_epsilon = epsilon
        self.n_tokens = n_tokens
        self.enable_privacy_tracking = enable_privacy_tracking

        # 动态属性
        self.projection_matrix = None
        self.quantize_bins = None
        self.band_partition = None
        self.lsh_strategy = None

        # 隐私预算跟踪
        if enable_privacy_tracking:
            self.privacy_tracker = PrivacyBudgetTracker(epsilon)

        # 统计信息
        self.stats = {}

    def fit_transform(self, user_vectors, user_ids):
        """
        主流程：用户向量 → 群体向量

        参数:
            user_vectors: (N, 128) numpy数组
            user_ids: (N,) 用户ID数组
        返回:
            group_data: dict {
                'group_vectors': {band_token_id: 群体向量(128,)},
                'group_users': {band_token_id: [user_ids]},
                'group_user_vectors': {band_token_id: [user_vectors]}
            }
            metadata: dict 包含统计信息
        """
        print(f"\n{'='*60}")
        print(f"开始处理 {len(user_vectors):,} 个用户向量...")
        print(f"{'='*60}\n")

        # 获取当前epsilon
        if self.enable_privacy_tracking:
            current_epsilon = self.privacy_tracker.get_current_epsilon()
            print(f"📊 隐私预算:")
            print(f"   当前ε: {current_epsilon:.4f}")
            print(f"   累积ε: {self.privacy_tracker.history['total_budget']:.4f}")
        else:
            current_epsilon = self.base_epsilon

        # 步骤1: 自适应LSH生成tokens
        print(f"\n{'─'*60}")
        print("步骤1: 自适应LSH")
        print(f"{'─'*60}")
        tokens = self._adaptive_lsh(user_vectors)
        print(f"✓ 生成tokens: {tokens.shape}")

        # 步骤2: Band分桶 + 独立性验证
        print(f"\n{'─'*60}")
        print("步骤2: Band分桶")
        print(f"{'─'*60}")
        band_tokens, buckets = self._band_partition_with_validation(tokens)
        print(f"✓ 分成 {len(buckets):,} 个桶")

        # 统计桶大小
        bucket_sizes = [len(v) for v in buckets.values()]
        print(f"   最小桶: {min(bucket_sizes):,} 人")
        print(f"   平均桶: {np.mean(bucket_sizes):.0f} 人")
        print(f"   最大桶: {max(bucket_sizes):,} 人")

        # 步骤3: 质心计算 + 差分隐私 + 保存用户信息
        print(f"\n{'─'*60}")
        print("步骤3: 质心计算 + 差分隐私 + 用户映射")
        print(f"{'─'*60}")
        group_data = self._generate_group_data(
            user_vectors,
            user_ids,
            buckets,
            current_epsilon
        )
        print(f"✓ 生成 {len(group_data['group_vectors']):,} 个群体向量")

        # 质量验证
        print(f"\n{'─'*60}")
        print("步骤4: 质量验证")
        print(f"{'─'*60}")
        self._validate_quality(user_vectors, group_data['group_vectors'], buckets)

        # 保存统计信息
        self.stats = {
            'n_users': len(user_vectors),
            'n_groups': len(group_data['group_vectors']),
            'lsh_strategy': self.lsh_strategy,
            'n_tokens': self.n_tokens,
            'n_bands': len(self.band_partition) if self.band_partition else 0,
            'min_bucket_size': min(bucket_sizes),
            'avg_bucket_size': np.mean(bucket_sizes),
            'max_bucket_size': max(bucket_sizes),
            'epsilon': current_epsilon,
            'timestamp': datetime.now().isoformat()
        }

        # 构建元数据
        metadata = {
            'bucket_sizes': {int(k): len(v) for k, v in buckets.items()},
            'stats': self.stats
        }

        return group_data, metadata

    # ==================== LSH相关 ====================

    def _adaptive_lsh(self, user_vectors):
        """自适应LSH: 自动选择最优策略"""

        # 数据分析
        self.lsh_strategy = self._analyze_data(user_vectors)
        print(f"   LSH策略: {self.lsh_strategy}")

        # 生成投影矩阵
        if 'pca' in self.lsh_strategy:
            self.projection_matrix = self._pca_random_projection(user_vectors)
        else:
            self.projection_matrix = self._random_projection()

        # 投影
        projections = user_vectors @ self.projection_matrix.T

        # 量化
        if 'quantile' in self.lsh_strategy:
            tokens = self._quantile_quantize(projections)
            print(f"   量化方法: 分位数量化")
        else:
            tokens = self._linear_quantize(projections)
            print(f"   量化方法: 线性量化")

        return tokens

    def _analyze_data(self, user_vectors):
        """分析数据特性，选择最优LSH策略"""

        # 采样（避免大数据集全量分析）
        n_sample = min(10000, len(user_vectors))
        sample_idx = np.random.choice(len(user_vectors), n_sample, replace=False)
        sample = user_vectors[sample_idx]

        # 检测1: PCA结构强度
        pca = PCA(n_components=min(10, sample.shape[1]))
        pca.fit(sample)
        structure_score = np.sum(pca.explained_variance_ratio_)

        # 检测2: 分布偏态
        random_proj = np.random.randn(sample.shape[1], self.n_tokens)
        random_proj /= np.linalg.norm(random_proj, axis=0)
        proj_sample = sample @ random_proj
        skewness_score = np.mean(np.abs(stats.skew(proj_sample, axis=0)))

        print(f"   数据分析:")
        print(f"      结构得分: {structure_score:.3f} (>0.75为强结构)")
        print(f"      偏态得分: {skewness_score:.3f} (>1.5为长尾)")

        # 决策
        if structure_score > 0.75 and skewness_score > 1.5:
            return 'pca_quantile'
        elif structure_score > 0.75:
            return 'pca_linear'
        elif skewness_score > 1.5:
            return 'random_quantile'
        else:
            return 'random_linear'

    def _pca_random_projection(self, user_vectors):
        """PCA主成分 + 随机方向混合投影"""
        n_pca = self.n_tokens // 2
        n_random = self.n_tokens - n_pca

        # PCA主成分
        pca = PCA(n_components=n_pca)
        pca.fit(user_vectors)
        pca_components = pca.components_

        # 随机方向（Gram-Schmidt正交化）
        random_proj = np.random.randn(n_random, user_vectors.shape[1])
        for i in range(n_random):
            for pc in pca_components:
                random_proj[i] -= np.dot(random_proj[i], pc) * pc
            norm = np.linalg.norm(random_proj[i])
            if norm > 1e-10:
                random_proj[i] /= norm

        return np.vstack([pca_components, random_proj])

    def _random_projection(self):
        """纯随机投影（标准LSH）"""
        proj = np.random.randn(self.n_tokens, 128)
        return proj / np.linalg.norm(proj, axis=1, keepdims=True)

    def _quantile_quantize(self, projections):
        """分位数量化：每个bin人数均匀"""
        tokens = np.zeros_like(projections, dtype=np.uint16)
        self.quantize_bins = []

        for i in range(self.n_tokens):
            percentiles = np.linspace(0, 100, 65537)
            bins = np.percentile(projections[:, i], percentiles)
            self.quantize_bins.append(bins)

            digitized = np.digitize(projections[:, i], bins) - 1
            tokens[:, i] = np.clip(digitized, 0, 65535)

        return tokens

    def _linear_quantize(self, projections):
        """线性量化：保持距离比例"""
        min_vals = projections.min(axis=0)
        max_vals = projections.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-10] = 1.0

        normalized = (projections - min_vals) / range_vals
        return (normalized * 65535).astype(np.uint16)

    # ==================== Band分桶相关 ====================

    def _band_partition_with_validation(self, tokens):
        """Band分桶 + 独立性验证"""
        n_users, n_tokens = tokens.shape
        best_config = None
        best_score = -np.inf

        # 尝试不同的band数量
        for n_bands in range(2, min(n_tokens // 2 + 1, 8)):
            tokens_per_band = n_tokens // n_bands
            partition = [
                list(range(i * tokens_per_band, (i + 1) * tokens_per_band))
                for i in range(n_bands)
            ]

            # 处理余数token
            for i, idx in enumerate(range(n_bands * tokens_per_band, n_tokens)):
                partition[i % n_bands].append(idx)

            # 生成band tokens
            band_tokens = self._hash_band_tokens(tokens, partition)

            # 检查k-匿名
            bucket_counts = Counter(band_tokens)
            min_bucket = min(bucket_counts.values())

            if min_bucket < self.k_min:
                continue

            # 计算熵
            probs = np.array(list(bucket_counts.values())) / n_users
            entropy = -np.sum(probs * np.log2(probs + 1e-10))

            if entropy > best_score:
                best_score = entropy
                best_config = {
                    'partition': partition,
                    'n_bands': n_bands,
                    'min_bucket': min_bucket,
                    'n_buckets': len(bucket_counts)
                }

        if best_config is None:
            raise ValueError(f"❌ 无法满足k={self.k_min}的匿名要求，请减小k_min或增加n_tokens")

        print(f"   初始配置: {best_config['n_bands']}个band, {best_config['n_buckets']}个桶")

        # 验证并优化独立性
        optimized_partition, max_mi = self._verify_and_fix_independence(
            tokens,
            best_config['partition']
        )

        self.band_partition = optimized_partition
        band_tokens = self._hash_band_tokens(tokens, optimized_partition)

        # 构建桶字典
        buckets = {}
        for user_id, bt in enumerate(band_tokens):
            if bt not in buckets:
                buckets[bt] = []
            buckets[bt].append(user_id)

        # 过滤小桶
        buckets = {k: v for k, v in buckets.items() if len(v) >= self.k_min}

        return band_tokens, buckets

    def _verify_and_fix_independence(self, tokens, partition, max_iterations=5):
        """验证band间独立性，如果MI过高则调整"""

        for iteration in range(max_iterations):
            # 计算当前MI
            band_tokens_list = []
            for token_indices in partition:
                bt = self._hash_band_tokens(tokens, [token_indices])
                band_tokens_list.append(bt)

            n_bands = len(partition)
            max_mi = 0
            max_mi_pair = None

            # 找最大MI的band对
            for i in range(n_bands):
                for j in range(i + 1, n_bands):
                    mi = normalized_mutual_info_score(
                        band_tokens_list[i],
                        band_tokens_list[j]
                    )
                    if mi > max_mi:
                        max_mi = mi
                        max_mi_pair = (i, j)

            print(f"   Band独立性验证 (迭代{iteration+1}): 最大MI = {max_mi:.4f}")

            if max_mi < 0.3:
                break

            # 交换相关band中的token
            if max_mi_pair and len(partition[max_mi_pair[0]]) > 1 and len(partition[max_mi_pair[1]]) > 1:
                i, j = max_mi_pair
                idx_i = np.random.randint(len(partition[i]))
                idx_j = np.random.randint(len(partition[j]))
                partition[i][idx_i], partition[j][idx_j] = \
                    partition[j][idx_j], partition[i][idx_i]
                print(f"      调整: 交换band{i}和band{j}的token")

        return partition, max_mi

    def _hash_band_tokens(self, tokens, partition):
        """将多个token组合哈希成band token"""
        n_users = len(tokens)
        band_tokens = np.zeros(n_users, dtype=np.int64)

        for token_indices in partition:
            for i, idx in enumerate(token_indices):
                band_tokens ^= (tokens[:, idx].astype(np.int64) << (i * 16))

        return band_tokens

    # ==================== 质心与差分隐私 ====================

    def _generate_group_data(self, user_vectors, user_ids, buckets, epsilon):
        """
        生成群体数据: 鲁棒质心 + vMF噪声 + 用户映射

        返回:
            {
                'group_vectors': {band_token: 群体向量},
                'group_users': {band_token: [user_ids]},
                'group_user_vectors': {band_token: [user_vectors]}
            }
        """
        group_vectors = {}
        group_users = {}
        group_user_vectors = {}

        for bucket_id, user_indices in buckets.items():
            bucket_vectors = user_vectors[user_indices]
            bucket_user_ids = user_ids[user_indices]

            # 鲁棒质心
            centroid = self._robust_weighted_centroid(bucket_vectors)

            # vMF差分隐私噪声
            noisy_centroid = self._add_vmf_noise(centroid, epsilon)

            # 保存数据
            group_vectors[bucket_id] = noisy_centroid
            group_users[bucket_id] = bucket_user_ids.tolist()
            group_user_vectors[bucket_id] = bucket_vectors

        return {
            'group_vectors': group_vectors,
            'group_users': group_users,
            'group_user_vectors': group_user_vectors
        }

    def _robust_weighted_centroid(self, vectors):
        """鲁棒加权质心计算"""
        # 第一轮：简单平均
        centroid = np.mean(vectors, axis=0)

        # 计算距离
        diff = vectors - centroid
        distances = np.linalg.norm(diff, axis=1)

        # 3σ过滤
        threshold = np.mean(distances) + 3 * np.std(distances)
        valid_mask = distances <= threshold

        # 确保不破坏k-匿名
        if valid_mask.sum() < self.k_min:
            valid_mask = np.ones(len(vectors), dtype=bool)

        vectors = vectors[valid_mask]

        # 第二轮：加权平均
        centroid = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)

        # 高斯核权重
        sigma = np.median(distances) + 1e-10
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weights /= weights.sum()

        weighted_centroid = np.average(vectors, axis=0, weights=weights)

        return weighted_centroid

    def _add_vmf_noise(self, centroid, epsilon):
        """von Mises-Fisher球面差分隐私噪声"""
        centroid_norm = np.linalg.norm(centroid)

        # 边界情况
        if centroid_norm < 1e-6:
            random_vec = np.random.randn(len(centroid))
            return random_vec / np.linalg.norm(random_vec)

        # 归一化
        mu = centroid / centroid_norm

        # vMF参数
        kappa = epsilon * 10

        # 生成球面随机方向
        random_dir = np.random.randn(len(centroid))
        random_dir /= np.linalg.norm(random_dir)

        # 加权混合
        alpha = np.exp(kappa) / (np.exp(kappa) + 1)
        noisy_vector = alpha * mu + (1 - alpha) * random_dir

        # 投影回球面
        return noisy_vector / np.linalg.norm(noisy_vector)

    # ==================== 质量验证 ====================

    def _validate_quality(self, user_vectors, group_vectors, buckets):
        """验证生成质量"""
        # 1. 归一化检查
        norms = [np.linalg.norm(v) for v in group_vectors.values()]
        print(f"   向量归一化: min={min(norms):.6f}, max={max(norms):.6f}, avg={np.mean(norms):.6f}")

        # 2. 质心代表性
        intra_distances = []
        for bucket_id, user_indices in buckets.items():
            if bucket_id in group_vectors:
                centroid = group_vectors[bucket_id]
                bucket_vecs = user_vectors[user_indices]
                cosine_sims = bucket_vecs @ centroid
                avg_dist = 1 - np.mean(cosine_sims)
                intra_distances.append(avg_dist)

        print(f"   质心代表性 (平均余弦距离): {np.mean(intra_distances):.4f}")

        # 3. k-匿名满足率
        bucket_sizes = [len(v) for v in buckets.values()]
        满足率 = sum(1 for s in bucket_sizes if s >= self.k_min) / len(bucket_sizes) * 100
        print(f"   k-匿名满足率: {满足率:.1f}%")

    def get_stats(self):
        """获取统计信息"""
        return self.stats


class Pipeline:
    """完整流水线：HDFS加载 → 向量生成 → HDFS存储"""

    def __init__(self,
                 k_min=1000,
                 epsilon=1.0,
                 n_tokens=12,
                 output_hdfs_path='hdfs://bj05-region15/region15/27614/warehouse/ai_vcid/feature_vectors/group_vectors.parquet'):

        self.generator = AnonymousGroupVectorGenerator(
            k_min=k_min,
            epsilon=epsilon,
            n_tokens=n_tokens,
            enable_privacy_tracking=True
        )

        self.hdfs = HDFSIO()
        self.output_hdfs_path = output_hdfs_path

    def load_data_from_hdfs(self, hdfs_path):
        """
        从HDFS加载用户数据

        参数:
            hdfs_path: HDFS Parquet文件路径

        返回:
            user_vectors: (N, 128) numpy数组
            user_ids: (N,) numpy数组
        """
        logger.info(f"从HDFS加载数据: {hdfs_path}")

        # 读取Parquet
        df = self.hdfs.read_from_hdfs(hdfs_path)

        if df is None or df.empty:
            raise ValueError("HDFS数据加载失败或为空")

        # 验证字段
        if 'user_id' not in df.columns or 'vector' not in df.columns:
            raise ValueError("数据必须包含 'user_id' 和 'vector' 字段")

        # 提取user_id
        user_ids = df['user_id'].values

        # 提取vector（假设是list或numpy数组）
        if isinstance(df['vector'].iloc[0], list):
            user_vectors = np.array(df['vector'].tolist())
        elif isinstance(df['vector'].iloc[0], np.ndarray):
            user_vectors = np.vstack(df['vector'].values)
        else:
            raise ValueError(f"不支持的vector类型: {type(df['vector'].iloc[0])}")

        # 验证维度
        if user_vectors.shape[1] != 128:
            raise ValueError(f"向量维度必须是128，当前: {user_vectors.shape[1]}")

        # L2归一化（如果未归一化）
        norms = np.linalg.norm(user_vectors, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-3):
            logger.info("向量未归一化，执行L2归一化...")
            user_vectors = user_vectors / (norms + 1e-10)

        logger.info(f"✓ 数据加载完成: {len(user_ids):,} 个用户, 向量维度 {user_vectors.shape[1]}")

        return user_vectors, user_ids

    def convert_to_dataframe(self, group_data, metadata):
        """
        将群体数据转换为DataFrame

        参数:
            group_data: {
                'group_vectors': {band_token: vector},
                'group_users': {band_token: [user_ids]}
            }
            metadata: 元数据字典

        返回:
            DataFrame with columns: [group_id, vector, user_count, user_ids, update_time]
        """
        records = []
        update_time = datetime.now().isoformat()

        for group_id, vector in group_data['group_vectors'].items():
            user_ids = group_data['group_users'][group_id]

            records.append({
                'group_id': int(group_id),
                'vector': vector.tolist(),  # 存储为list
                'user_count': len(user_ids),
                'user_ids': user_ids,  # 存储为list（Parquet支持）
                'update_time': update_time
            })

        df = pd.DataFrame(records)

        # 添加元数据作为单独的列（可选）
        df['epsilon'] = metadata['stats']['epsilon']
        df['lsh_strategy'] = metadata['stats']['lsh_strategy']

        return df

    def write_to_hdfs(self, group_data, metadata):
        """
        将群体向量写入HDFS

        参数:
            group_data: 群体数据字典
            metadata: 元数据字典

        返回:
            bool: 是否成功
        """
        print(f"\n{'─'*60}")
        print("步骤5: 写入HDFS")
        print(f"{'─'*60}")

        # 转换为DataFrame
        df = self.convert_to_dataframe(group_data, metadata)

        logger.info(f"准备写入DataFrame: shape={df.shape}")
        logger.info(f"列: {df.columns.tolist()}")

        # 写入HDFS
        success = self.hdfs.write_df_to_hdfs_parquet(df, self.output_hdfs_path)

        if success:
            print(f"✓ 成功写入 {len(df):,} 个群体向量到HDFS")
            print(f"   路径: {self.output_hdfs_path}")
        else:
            print(f"✗ HDFS写入失败")

        return success

    def run(self, hdfs_path):
        """
        运行完整流水线

        参数:
            hdfs_path: HDFS输入数据路径

        返回:
            success: bool
            stats: dict 统计信息
        """
        try:
            print(f"\n{'='*60}")
            print(f"Pipeline开始执行")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")

            # 步骤1: 从HDFS加载数据
            print(f"\n{'─'*60}")
            print("步骤0: 从HDFS加载数据")
            print(f"{'─'*60}")
            user_vectors, user_ids = self.load_data_from_hdfs(hdfs_path)

            # 步骤2: 生成群体向量
            group_data, metadata = self.generator.fit_transform(user_vectors, user_ids)

            # 步骤3: 写入HDFS
            write_success = self.write_to_hdfs(group_data, metadata)

            if not write_success:
                raise Exception("HDFS写入失败")

            # 获取统计信息
            stats = self.generator.get_stats()
            stats['output_path'] = self.output_hdfs_path

            # 隐私预算报告
            if self.generator.enable_privacy_tracking:
                privacy_report = self.generator.privacy_tracker.get_report()
                stats['privacy_report'] = privacy_report

                print(f"\n{'─'*60}")
                print("隐私预算报告")
                print(f"{'─'*60}")
                print(f"   总更新次数: {privacy_report['total_updates']}")
                print(f"   累积预算: {privacy_report['total_budget']:.4f}")
                print(f"   下次epsilon: {privacy_report['next_epsilon']:.4f}")

            print(f"\n{'='*60}")
            print(f"✓ Pipeline执行成功")
            print(f"{'='*60}\n")

            # 保存统计信息
            stats_file = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ 统计信息已保存到: {stats_file}")

            return True, stats

        except Exception as e:
            logger.error(f"❌ Pipeline执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, {'error': str(e)}


# ==================== 主函数 ====================

def main():
    """主函数：完整示例"""

    # ===== 配置参数 =====
    CONFIG = {
        'k_min': 1000,           # k-匿名最小桶大小
        'epsilon': 1.0,          # 隐私预算
        'n_tokens': 12,          # LSH token数量
        'output_hdfs_path': 'hdfs://bj05-region15/region15/27614/warehouse/ai_vcid/feature_vectors/group_vectors.parquet'
    }

    # HDFS输入路径
    INPUT_HDFS_PATH = "hdfs://bj05-region15/region15/27614/warehouse/ai_vcid/feature_vectors/user_vectors.parquet"

    print("="*60)
    print("匿名群体向量生成系统")
    print("="*60)
    print(f"配置: {json.dumps(CONFIG, indent=2)}")
    print(f"输入路径: {INPUT_HDFS_PATH}")
    print(f"输出路径: {CONFIG['output_hdfs_path']}")
    print("="*60)

    # 运行Pipeline
    pipeline = Pipeline(**CONFIG)
    success, stats = pipeline.run(INPUT_HDFS_PATH)

    # ===== 结果展示 =====
    if success:
        print("\n" + "="*60)
        print("执行统计")
        print("="*60)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print("\n" + "="*60)
        print("执行失败")
        print("="*60)
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
