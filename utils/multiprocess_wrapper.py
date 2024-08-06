import torch.multiprocessing as multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time


def multiprocess_wrapper(func, args_list, logger, n_processes, n_workers=-1, log_gap=100):
    """
    Multiprocess wrapper for function calls.
    Use multiprocess to accelerate the process of data processing.
    n_workers is limited by computing resources.

    Args:
        func: function to be called
        args_list: list of arguments for the function
        logger: logger
        n_processes: number of processes to process data
        n_workers: number of workers to run in parallel
        log_gap: log gap

    Returns:
        list of results

    """
    multiprocessing.set_start_method('spawn', force=True)

    if n_workers == -1:
        n_workers = n_processes
    assert len(args_list) == n_processes, \
        f'number of processes {n_processes} does not match number of arguments {len(args_list)}'
    assert n_workers <= n_processes, "n_workers must be less than or equal to n_processes"

    logger.info(f'using {n_processes} processes and {n_workers} workers')

    results = []

    with multiprocessing.Manager() as manager:
        _progress = manager.dict()

        for offset in range(0, n_processes, n_workers):
            process_list = []
            args_list_batch = args_list[offset:offset + n_workers]
            logger.info("Starting batch {} to {}".format(offset, offset + len(args_list_batch)))

            with ProcessPoolExecutor(max_workers=len(args_list_batch)) as executor:

                for i in range(offset, offset + len(args_list_batch)):
                    process_list.append(executor.submit(func, _progress, i, *args_list[i]))

                next_log = log_gap
                while n_finished := sum([process.done() for process in process_list]) < len(process_list):
                    latest_list = []
                    total_list = []
                    for task_id, progress_dict in _progress.items():
                        if task_id <= offset:
                            continue
                        latest = progress_dict["latest"]
                        total = progress_dict["total"]
                        latest_list.append(latest)
                        total_list.append(total)
                    if sum(latest_list) >= next_log:
                        logger.info("***** Batch {} Progress: {}/{} ({:.2f}%) ******".
                                    format(offset // n_workers, sum(latest_list), sum(total_list),
                                           sum(latest_list) / sum(total_list) * 100))
                        for i, (l, t) in enumerate(zip(latest_list, total_list)):
                            logger.info("  Task {} Progress: {}/{} ({:.2f}%)".format(i+offset, l, t, l / t * 100))

                        next_log += log_gap
                    time.sleep(5)
                for process in process_list:
                    results.append(process.result())

    return results
