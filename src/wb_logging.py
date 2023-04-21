from mpi4py import MPI

__HAS_WB: bool = False
try:
    import wandb
    # Sue me
    __LOGS = []
    __LOG_COMM: MPI.Comm
    __ITERATION_DATA: dict = {}
except:
    # Sue me
    __HAS_WB = False


def set_iteration_data(data: dict):
    """
    This data will be added to any subsequent call to log before the next call to set_iteration_data
    :param data:
    :return:
    """
    if __HAS_WB:
        global __ITERATION_DATA
        __ITERATION_DATA = data.copy()


def log(data: dict):
    if __HAS_WB:
        data.update(__ITERATION_DATA)
        __LOGS.append(data)


def finish():
    if __HAS_WB:

        data = __LOGS
        all_data = __LOG_COMM.gather(data, 0)

        if __LOG_COMM.Get_rank() == 0:
            for i, log in enumerate(all_data):
                for item in log:
                    item['rank'] = i
                    wandb.log(item)

            wandb.finish()

        # Wait for root...
        __LOG_COMM.Barrier()

def wandb_init(comm: MPI.Comm,
               dataset,
               n_features,
               iterations,
               device,
               algorithm,
               block_width,
               wandb_api_key: str = None
               ):
    """
    Initializes a wandb logging run.
    If None is returned, something went wrong and you should not log anything.
    :param comm:
    :param dataset:
    :param n_features:
    :param iterations:
    :param device:
    :param algorithm:
    :param block_width:
    :param wandb_api_key:
    :return:
    """
    if wandb_api_key is not None:
        try:
            import platform
            import wandb

            has_wb = False
            # start a new wandb run to track this script
            if comm.Get_rank() == 0:
                wandb.login(key=wandb_api_key)
                #print("LOGIN")
                has_wb = True

                dataset = (dataset.split('/'))[-1]

                wandb.init(
                    # set the wandb project where this run will be logged
                    project="spmm-mpi",
                    # track hyperparameters and run metadata
                    config={
                        "dataset": dataset,
                        "width": block_width,
                        "n_features": n_features,
                        "iterations": iterations,
                        "device": device,
                        "ranks": comm.Get_size(),
                        "host": platform.node(),
                        "algorithm": algorithm,
                    },
                    reinit=True,
                    tags=[algorithm, device, dataset]
                )

            has_wb = comm.bcast(has_wb, root=0)
            global __HAS_WB
            __HAS_WB = has_wb
            #print("HAS_WB", __HAS_WB, has_wb)
            if __HAS_WB:
                global __LOG_COMM
                __LOG_COMM = comm

        except:
            wandb_api_key = None
            __HAS_WB = False
            print("A WANDB EXCEPTION HAS OCCURRED")


    return wandb_api_key

