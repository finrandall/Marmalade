import os
from operator import index

from numba import config, set_num_threads


_BLAS_LIMITER = None


def available_cpus():
    process_count = getattr(os, "process_cpu_count", lambda: None)()
    if hasattr(os, "sched_getaffinity"):
        affinity_count = len(os.sched_getaffinity(0))
        process_count = min(process_count or affinity_count, affinity_count)
    return max(1, process_count or os.cpu_count() or 1)


def local_mpi_ranks(comm=None, mpi=None):
    if comm is not None and mpi is not None:
        local_comm = comm.Split_type(mpi.COMM_TYPE_SHARED,
                                     key=comm.Get_rank())
        try:
            return local_comm.Get_size()
        finally:
            local_comm.Free()
    for name in ("OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS",
                 "MV2_COMM_WORLD_LOCAL_SIZE", "PMIX_LOCAL_SIZE"):
        if name in os.environ:
            try:
                return max(1, int(os.environ[name]))
            except ValueError:
                pass
    return 1


def _positive_integer(value, name):
    try:
        converted = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if converted < 1:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def resolve_runtime_settings(settings=None, local_ranks=1, cpu_count=None):
    settings = {} if settings is None else settings
    local_ranks = _positive_integer(local_ranks, "local_ranks")
    cpu_count = available_cpus() if cpu_count is None else _positive_integer(
        cpu_count, "cpu_count"
    )
    budget = cpu_count // local_ranks
    if budget < 1:
        raise ValueError(
            f"{local_ranks} local MPI ranks exceed the {cpu_count} available "
            f"CPU threads; launch at most {cpu_count} ranks on this host"
        )
    defaults = {
        "threads_per_rank": min(4, budget, config.NUMBA_NUM_THREADS),
        "fft_workers": min(4, budget),
        "blas_threads": 1,
    }
    resolved = {}
    for name, default in defaults.items():
        value = settings.get(name)
        resolved[name] = _positive_integer(
            default if value is None else value, name
        )
        if resolved[name] > budget:
            raise ValueError(
                f"{name}={resolved[name]} with {local_ranks} local MPI ranks "
                f"exceeds the {cpu_count}-thread CPU budget; use at most "
                f"{budget} {name} per rank or launch fewer ranks"
            )
    if resolved["threads_per_rank"] > config.NUMBA_NUM_THREADS:
        raise ValueError(
            f"threads_per_rank must not exceed the Numba thread limit "
            f"{config.NUMBA_NUM_THREADS}"
        )
    return {
        "local_ranks": local_ranks,
        "cpu_count": cpu_count,
        "numba_threads": resolved["threads_per_rank"],
        "fft_workers": resolved["fft_workers"],
        "blas_threads": resolved["blas_threads"],
    }


def configure_runtime(settings=None, comm=None, mpi=None):
    global _BLAS_LIMITER
    runtime = resolve_runtime_settings(
        settings, local_ranks=local_mpi_ranks(comm, mpi)
    )
    set_num_threads(runtime["numba_threads"])
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        raise RuntimeError(
            "install the shared ASD dependencies, including threadpoolctl, "
            "to control BLAS threads"
        ) from None
    else:
        if _BLAS_LIMITER is not None:
            _BLAS_LIMITER.restore_original_limits()
        _BLAS_LIMITER = threadpool_limits(
            limits=runtime["blas_threads"], user_api="blas"
        )
    return runtime
