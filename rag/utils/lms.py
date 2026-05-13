# -*- coding: utf-8 -*-
"""
LM Studio 自动化控制模块

提供 LM Studio 本地 API 服务的启动/停止、模型加载/卸载等自动化控制功能。
所有模型名称和 GPU 配置均从 config.py 统一读取。

Public API:
    load_both_models()      -- 加载 LLM + Embedding 双模型 (供 chat_rag 使用)
    unload_models()          -- 卸载所有模型并停止服务 (供 chat_rag 使用)
    load_model(model_type)   -- 加载指定类型的模型
    unload_all()             -- 卸载所有模型
    start_server()           -- 启动 LM Studio API 服务
    stop_server()            -- 停止 LM Studio API 服务
"""

import subprocess
import sys
import time
import logging
from typing import Literal

import config

_log = logging.getLogger(__name__)

ModelType = Literal["llm", "vlm", "embedding"]


def _run_lms(*args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    """执行 lms CLI 命令，统一错误处理。"""
    cmd = ["lms"] + list(args)
    _log.debug("执行命令: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)


def start_server() -> None:
    """启动 LM Studio API 服务 (端口 1234)。"""
    _log.info("正在启动 LM Studio API 服务...")
    try:
        _run_lms("server", "start")
        time.sleep(2)
        _log.info("API 服务已启动")
    except Exception as e:
        _log.error("API 服务启动失败: %s", e)
        raise


def stop_server() -> None:
    """停止 LM Studio API 服务并释放端口。"""
    _log.info("正在停止 LM Studio API 服务...")
    try:
        _run_lms("server", "stop", check=False)
        time.sleep(1)
        _log.info("API 服务已停止")
    except Exception as e:
        _log.warning("停止服务时出现警告: %s", e)


def unload_all() -> None:
    """卸载所有已加载的模型，释放 GPU 显存。"""
    _log.info("正在卸载所有模型...")
    try:
        _run_lms("unload", "--all", check=False)
        time.sleep(2)
        _log.info("所有模型已卸载")
    except Exception as e:
        _log.warning("卸载模型时出现警告: %s", e)


def _load_model_lms(model_type: ModelType, gpu_ratio: str | None = None) -> None:
    """执行 lms load 命令（不处理 server/unload，由调用方管理）。"""
    model_map = {
        "llm":       (config.LLM_MODEL_NAME,       config.LLM_CONTEXT_LENGTH),
        "vlm":       (config.VLM_MODEL_NAME,       None),
        "embedding": (config.EMBEDDING_MODEL_NAME, config.EMBEDDING_CONTEXT_LENGTH),
    }

    if model_type not in model_map:
        raise ValueError(f"未知模型类型: {model_type}，可选: 'llm', 'vlm', 'embedding'")

    model_name, ctx_length = model_map[model_type]
    if gpu_ratio is None:
        gpu_ratio = config.GPU_OFFLOAD[model_type]

    _log.info("正在加载 %s 模型: %s (GPU: %s)", model_type, model_name, gpu_ratio)

    load_args = ["lms", "load", model_name, "--gpu", str(gpu_ratio)]
    if ctx_length is not None:
        load_args.extend(["-c", str(ctx_length)])

    try:
        subprocess.run(load_args, check=True, capture_output=True, text=True)
        time.sleep(3)
        _log.info("模型 %s 加载成功", model_name)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "未知 CLI 错误"
        _log.error("模型加载失败: %s", error_msg)
        raise RuntimeError(f"加载 {model_name} 失败: {error_msg}") from e
    except FileNotFoundError:
        _log.error("找不到 'lms' 命令，请确保 LM Studio CLI 已安装")
        raise


def load_model(model_type: ModelType, gpu_ratio: str | None = None) -> None:
    """
    加载单个模型（先卸载其他模型再加载）。

    Args:
        model_type: 模型类型 — 'llm' (推理), 'vlm' (视觉), 'embedding' (向量)
        gpu_ratio: GPU 显存分配比例，为 None 时使用 config.GPU_OFFLOAD 的默认值
    """
    start_server()
    unload_all()
    _load_model_lms(model_type, gpu_ratio)


def load_both_models() -> None:
    """
    加载 LLM 推理模型 + Embedding 向量模型。

    显存分配比例由 config.GPU_OFFLOAD 统一管理。
    供 chat_rag.py 在启动对话前调用。
    """
    print("\n正在加载大模型，请稍候...")
    print(f"  [1/2] 加载嵌入模型: {config.EMBEDDING_MODEL_NAME}")
    _log.info("正在加载双模型引擎...")
    start_server()
    unload_all()
    _load_model_lms("embedding")
    print(f"  [2/2] 加载推理模型: {config.LLM_MODEL_NAME}")
    _load_model_lms("llm")
    print("大模型加载完成，可以开始对话。")
    _log.info("双模型引擎加载完成")


def unload_models() -> None:
    """
    卸载所有模型并停止 API 服务。

    供 chat_rag.py 在对话结束后调用，释放 GPU 资源。
    """
    print("\n正在释放 GPU 资源...")
    _log.info("正在释放 GPU 资源...")
    unload_all()
    stop_server()
    print("GPU 资源已释放，再见。")
    _log.info("GPU 资源已释放")
