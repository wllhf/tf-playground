import os


def mkrundir(log_dir='./log'):
    run = 0
    while os.path.exists(os.path.join(log_dir, "run%s" % run)):
        run += 1
    run_dir = os.path.join(log_dir, "run%s" % run)
    os.makedirs(run_dir)
    return run_dir
