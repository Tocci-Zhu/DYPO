# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import pickle
import os
from verl.utils.device import is_cuda_available
import socket

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from .dapo_ray_trainer import RayDAPOTrainer

from omegaconf import OmegaConf

# import os
# os.environ['WANDB_X_EXECUTABLE'] = '/usr/bin/python3'


class RAGKnowledgeBase:
    """RAG知识库类，用于检索相似的问答对"""
    
    def __init__(self, knowledge_base_path="/mnt/tenant-home_speed/xdy/others/livedata/livedata_for_train-processed_output.jsonl"):
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.cache_path = knowledge_base_path + ".rag_cache.pkl"
        
        self._load_knowledge_base()
        self._build_index()
    
    def _load_knowledge_base(self):
        """加载知识库"""
        print(f"正在加载知识库: {self.knowledge_base_path}")
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.knowledge_base.append(data)
        print(f"知识库加载完成，共 {len(self.knowledge_base)} 条记录")
    
    def _preprocess_text(self, text):
        """文本预处理，使用jieba分词"""
        if not text:
            return ""
        # 使用jieba分词
        words = jieba.cut(text)
        return " ".join(words)
    
    def _build_index(self):
        """构建TF-IDF索引"""
        # 检查是否存在缓存
        if os.path.exists(self.cache_path):
            print("正在加载RAG缓存...")
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.vectorizer = cache_data['vectorizer']
                self.tfidf_matrix = cache_data['tfidf_matrix']
            print("RAG缓存加载完成")
            return
        
        print("正在构建TF-IDF索引...")
        # 提取所有问题文本进行预处理
        questions = []
        for item in self.knowledge_base:
            question = item.get('concept_based_question', '')
            questions.append(self._preprocess_text(question))
        
        # 构建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words=None,  # 中文不使用英文停用词
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # 构建TF-IDF矩阵
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)
        
        # 保存缓存
        cache_data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print("TF-IDF索引构建完成")
    
    def retrieve_similar(self, query, top_k=5):
        """检索最相似的问答对"""
        if not query:
            return []
        
        # 预处理查询文本
        processed_query = self._preprocess_text(query)
        
        # 向量化查询
        query_vector = self.vectorizer.transform([processed_query])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 获取最相似的top_k个结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回有相似度的结果
                item = self.knowledge_base[idx].copy()
                item['similarity'] = float(similarities[idx])
                results.append(item)
        
        return results
    
    def format_rag_context(self, similar_items, tokenizer=None, max_rag_tokens=512):
        """格式化RAG上下文为messages格式，并控制长度"""
        if not similar_items:
            return []
        
        rag_messages = []
        current_tokens = 0
        
        # 添加系统消息说明参考内容
        system_msg = {
            "role": "system", 
            "content": "以下是相关参考问答："
        }
        rag_messages.append(system_msg)
        
        if tokenizer:
            current_tokens += len(tokenizer.encode(system_msg["content"]))
        
        for i, item in enumerate(similar_items, 1):
            question = item.get('original_question', '')
            answer = item.get('concept_based_answer', '')
            
            # 截断过长的答案
            if len(answer) > 200:
                answer = answer[:200] + "..."
            
            user_content = f"Q{i}: {question}"
            assistant_content = f"A{i}: {answer}"
            
            # 估算token数量（粗略估计：中文1个字符约1个token，英文按空格分割）
            if tokenizer:
                user_tokens = len(tokenizer.encode(user_content))
                assistant_tokens = len(tokenizer.encode(assistant_content))
                
                # 检查是否会超过限制
                if current_tokens + user_tokens + assistant_tokens > max_rag_tokens:
                    break
                
                current_tokens += user_tokens + assistant_tokens
            
            # 添加参考问题和答案
            rag_messages.append({
                "role": "user",
                "content": user_content
            })
            
            rag_messages.append({
                "role": "assistant", 
                "content": assistant_content
            })
        
        # 添加分隔消息
        separator_msg = {
            "role": "system",
            "content": "现在请回答当前问题："
        }
        rag_messages.append(separator_msg)
        
        return rag_messages


# 全局RAG知识库实例
rag_kb = None


def get_rag_knowledge_base():
    """获取RAG知识库实例（单例模式）"""
    global rag_kb
    if rag_kb is None:
        rag_kb = RAGKnowledgeBase()
    return rag_kb


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and OmegaConf.select(config.global_profiler, "steps") is not None
        and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none


        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset with RAG enhancement.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset with RAG enhancement.
    """
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # 检查是否启用RAG
    enable_rag = data_config.get("enable_rag", True)
    
    if enable_rag:
        print("启用RAG功能，使用RAGEnhancedRLHFDataset")
        if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
            from verl.utils.import_utils import load_extern_type
            base_dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
            if not issubclass(base_dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
        else:
            base_dataset_cls = RLHFDataset
        
        # 创建RAG增强的数据集类
        class RAGEnhancedRLHFDataset(base_dataset_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rag_kb = get_rag_knowledge_base()
                self.rag_top_k = data_config.get("rag_top_k", 5)
                self.tokenizer = tokenizer # 将tokenizer传递给数据集类
                self.max_prompt_length = data_config.get("max_prompt_length", 2048) # 从配置中获取最大长度
                print(f"RAG增强数据集初始化完成，top_k={self.rag_top_k}")
            
            def _build_messages(self, example: dict):
                """重写_build_messages方法，在构建messages时加入RAG检索结果"""
                # 先调用父类方法获取原始messages
                original_messages = super()._build_messages(example)
                
                # 提取用户问题用于RAG检索
                query_text = ""
                if original_messages:
                    last_message = original_messages[-1]
                    if last_message.get("role") == "user":
                        content = last_message.get("content", "")
                        if isinstance(content, str):
                            query_text = content
                        elif isinstance(content, list):
                            # 处理多模态内容，提取文本部分
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                            query_text = " ".join(text_parts)
                
                if query_text:
                    # 检索相似的问答对
                    similar_items = self.rag_kb.retrieve_similar(query_text, top_k=self.rag_top_k)
                    
                    if similar_items:
                        # 计算剩余可用的token数量
                        max_rag_tokens = max(200, self.max_prompt_length * 0.75)  # 预留3/4给RAG内容
                        
                        # 格式化RAG上下文为messages，控制长度
                        rag_messages = self.rag_kb.format_rag_context(
                            similar_items, 
                            tokenizer=self.tokenizer,
                            max_rag_tokens=max_rag_tokens
                        )
                        
                        # 将RAG消息插入到原始messages之前
                        enhanced_messages = []
                        system_messages = []
                        user_assistant_messages = []
                        
                        for message in original_messages:
                            if message.get("role") == "system":
                                system_messages.append(message)
                            else:
                                user_assistant_messages.append(message)
                        
                        # 组合消息：系统消息 + RAG消息 + 用户对话消息
                        enhanced_messages.extend(system_messages)
                        enhanced_messages.extend(rag_messages)
                        enhanced_messages.extend(user_assistant_messages)
                        
                        return enhanced_messages
                
                return original_messages
        
        dataset_cls = RAGEnhancedRLHFDataset
    else:
        print("未启用RAG功能，使用标准数据集")
        if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
            from verl.utils.import_utils import load_extern_type
            dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset

    print(f"使用数据集类: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
