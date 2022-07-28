import argparse
import os
import uuid
from pathlib import Path

import main as classification
import submitit
import yaml

def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for HorNet", parents=[classification_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=72, type=int, help="Duration of the job, in hours")
    parser.add_argument("--job_name", default="gnnet", type=str, help="Job name")
    parser.add_argument("--job_dir", default="", type=str, help="Job directory; leave empty for default")
    parser.add_argument("--partition", default="a100", type=str, help="Partition where to submit")
    # parser.add_argument("--use_volta32", action='store_true', default=True, help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    parser.add_argument("--debug", default=False, action='store_true', help="whether debug on mbzuai")
    return parser.parse_args()

def get_init_file(path):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(path), exist_ok=True)
    init_file = path / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(Path(self.args.job_dir)).as_uri()
        self.args.auto_resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        # self.args.output_dir = Path(self.args.job_dir)
        self.args.slurm_job_id = job_env.job_id
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    with open("constants.yaml", "r") as f:
        constants = yaml.safe_load(f)
    
    args = parse_args()    
    if args.job_dir == "":
        args.job_dir = constants["experiment_folder"]
    else:
        raise NotImplementedError

    if args.debug:
        args.job_dir = constants["debug_experiment_folder"]

    slurm_folder = constants["slurm_folder"]

    executor = submitit.AutoExecutor(folder=os.path.join(args.job_dir, slurm_folder), slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout * 60

    partition = args.partition
    kwargs = {}
    # if args.use_volta32:
    #     kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        # mem_gb=50 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        # timeout_min=0,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        **kwargs
    )

    executor.update_parameters(name=args.job_name)

    args.dist_url = get_init_file(Path(args.job_dir)).as_uri()
    args.output_dir = os.path.join(args.job_dir, args.exp_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

    all_jobids_filename = os.path.join(args.job_dir, "all_jobids.txt")
    with open(all_jobids_filename, "a") as fd:
        fd.write(f"{job.job_id}\n")

if __name__ == "__main__":
    main()