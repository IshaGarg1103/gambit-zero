"""Minimal explicit W&B tracking. Credentials are read from the environment only."""
import os
from pathlib import Path


class Tracker:
    def __init__(self, config, directory, *, mode='online', run_id=None):
        if mode not in ('online','offline','disabled'):raise ValueError('Invalid W&B mode')
        self.run=None
        if mode=='disabled':return
        import wandb
        if mode=='online' and not os.environ.get('WANDB_API_KEY'):
            raise RuntimeError('WANDB_API_KEY must be set in environment (use Molab Secrets)')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        # All recorded config is caller-selected scientific metadata, never env dumps.
        self.run=wandb.init(entity='ishagarg-research',project='gambit-zero',
            name='stage3a-fullbatch-1024-confirmation',job_type='single-fullbatch-confirmation',
            group='m2-overfit-diagnostics',
            id=run_id,resume='allow' if run_id else None,mode=mode,dir=str(directory),
            config=config,tags=['stage3a','25m','history-dropout-0'],
            settings=wandb.Settings(disable_git=True,save_code=False,disable_code=True,console='off'))
    @property
    def url(self):return self.run.url if self.run else None
    @property
    def id(self):return self.run.id if self.run else None
    def log(self,metrics,step):
        if self.run:self.run.log(metrics,step=step)
    def checkpoint(self,directory,receipt):
        if not self.run:return None
        import wandb
        artifact=wandb.Artifact('stage3a-fullbatch-confirmation-checkpoint',type='checkpoint',
            metadata={'step':receipt['step'],'sha256':receipt['sha256']})
        artifact.add_file(str(Path(directory)/receipt['file']))
        artifact.add_file(str(Path(directory)/'checkpoint.json'))
        self.run.log_artifact(artifact)
        return artifact.name
    def finish(self,success):
        if self.run:self.run.finish(exit_code=0 if success else 1)
