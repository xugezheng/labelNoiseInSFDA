import logging
import time
import os.path as osp


__all__ = ["TxtLogger"]


def TxtLogger(filename, verbosity="info", logname="sfda"):
    level_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        # "critical": logging.CRITICAL,
    }
    formatter = logging.Formatter(
        fmt=
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(logname)
    logger.setLevel(level_dict[verbosity])
    # file handler
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, fh, sh


def set_logger(prm):
    now = int(round(time.time() * 1000))
    log_file = f"{prm.key_info}_{prm.dset}_{prm.source[0]}_{prm.target[0]}_seed_{prm.seed}_{str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000)))}.log"
    log_dir = osp.abspath(
            osp.join(
                prm.log_dir,
                prm.expname,
                prm.dset,
                prm.name,
            ))
    return log_dir, log_file






    
