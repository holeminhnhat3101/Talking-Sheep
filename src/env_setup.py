"""Thiết lập biến môi trường cho việc quản lý luồng trước khi import các thư viện nặng."""

import os
import logging

try:
    from .config import OMP_NUM_THREADS_VAL, OPENBLAS_NUM_THREADS_VAL
except ImportError:
    from src.config import OMP_NUM_THREADS_VAL, OPENBLAS_NUM_THREADS_VAL

logger = logging.getLogger(__name__)

def setup_thread_env():
    """Thiết lập các biến môi trường để giới hạn pool luồng của các thư viện toán học."""
    # Các biến này cần được đặt TRƯỚC khi numpy, torch, onnxruntime hoặc llama-cpp được import
    os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS_VAL)
    os.environ["OPENBLAS_NUM_THREADS"] = str(OPENBLAS_NUM_THREADS_VAL)
    os.environ["MKL_NUM_THREADS"] = str(OPENBLAS_NUM_THREADS_VAL)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(OPENBLAS_NUM_THREADS_VAL)
    os.environ["NUMEXPR_NUM_THREADS"] = str(OMP_NUM_THREADS_VAL)

    logger.info(
        "Thread environment set: OMP=%s, OpenBLAS=%s",
        OMP_NUM_THREADS_VAL,
        OPENBLAS_NUM_THREADS_VAL
    )

# Tự động thực hiện khi module được import
setup_thread_env()
