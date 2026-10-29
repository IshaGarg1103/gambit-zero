"""Month-at-a-time bridge: verified source -> HF -> cloud-backed second build."""
import hashlib
import json
from pathlib import Path

import yaml

from tinychess.data.archive_scan import DiskGuard,ScanError,canonical
from tinychess.data.hf_audit import AuditIndex
from tinychess.data.hf_card import publish_docs
from tinychess.data.hf_month import publish_month,reproduce_month,run_evaluation_inputs
from tinychess.data.hf_upload import VerifiedUploader
from tinychess.data.sources import sha256_file


class HfMonthlyDestination:
    def __init__(self,config,deadline=None):
        self.config=config
        self.hf=yaml.safe_load(Path(config['destination']['config']).read_text())
        if not self.hf['full_scan_enabled']:
            raise ScanError('HF full-scan mode remains explicitly disabled')
        self.run_id=self.hf['run_id']
        self.frozen,self.d_sha=run_evaluation_inputs(config,config['frozen_dataset_d_manifest'],synthetic=False)
        signature=hashlib.sha256(canonical(dict(seed=config['seed'],months=config['months'],
            filters=config['filters'],dataset_d=self.d_sha,run_id=self.run_id,repo=self.hf['repo_id'])).encode()).hexdigest()
        self.uploader=VerifiedUploader(self.hf['repo_id'],self.hf['spool'],private=self.hf['private'],
            spool_cap_bytes=self.hf['spool_cap_bytes'],reserve_disk_bytes=self.hf['free_disk_reserve_bytes'],deadline=deadline,
            transfer_budget_bytes=self.hf.get('transfer_budget_bytes'))
        # Fail authentication/visibility BEFORE spending archive bandwidth.
        try:
            self.uploader._transport().ensure_repo()
        except Exception as exc:
            from tinychess.data.hf_upload import UploadError
            if isinstance(exc,UploadError):raise
            raise UploadError('Hub authorization/visibility check failed ('+type(exc).__name__+')') from None
        guard=DiskGuard(self.hf['audit'],self.hf['audit_cap_bytes'],self.hf['free_disk_reserve_bytes'])
        self.audit=AuditIndex(self.hf['audit'],signature,guard)
        self.completed={}
        self.reproduced=set()
        journal_path=self.uploader.root/'upload_journal.json'
        journal=json.loads(journal_path.read_text()).get('files',{}) if journal_path.exists() else {}
        for month in config['months']:
            path=self.uploader.root/f'{self.run_id}-{month}.complete.json'
            base=f'data/{self.run_id}/manifests/{month}'
            receipt=journal.get(base+'.json',{})
            reproduced=journal.get(base+'.reproduction.json',{})
            if (path.exists() and receipt.get('status') in ['verified','deleted']
                    and reproduced.get('status') in ['verified','deleted']):
                if sha256_file(path)!=receipt['sha256']:
                    raise ScanError('Retained month manifest checksum mismatch')
                complete=json.loads(path.read_text())
                if complete['dataset_d_sha256']!=self.d_sha:
                    raise ScanError('Retained month uses a different evaluation policy')
                self.completed[month]=complete
                self.reproduced.add(month)

    def resume_completed(self,source):
        month=source['month']
        path=self.uploader.root/f'{self.run_id}-{month}.complete.json'
        if not path.exists():return False
        complete=json.loads(path.read_text())
        if (complete['source']['published_sha256']!=source['published_sha256']
                or complete['accepted_games']!=self.config['months'][month]['accepted_games']
                or complete['dataset_d_sha256']!=self.d_sha):
            raise ScanError('Completed HF month disagrees with current frozen run')
        self.uploader.publish(path,f'data/{self.run_id}/manifests/{month}.json',delete_after_verify=False)
        # __init__ has already checked retained manifest integrity and verified
        # upload/reproduction receipts. A catalog-only retry can reuse them.
        if month not in self.reproduced:
            self._second_build(complete)
        return True

    def publish_checkpoint(self,path):
        path=Path(path)
        before=sha256_file(path)
        result=publish_month(path,self.config,self.config['frozen_dataset_d_manifest'],self.uploader,self.audit,
            run_id=self.run_id,max_file_bytes=self.hf['max_shard_bytes'],max_games=self.hf['max_games_per_shard'])
        self._second_build(result['complete'])
        self._catalog()
        # Only the validated candidate checkpoint is retired, AFTER both builds
        # and the selected-PGN/month manifests have verified cloud receipts.
        # The selected PGNs remain remotely available for deterministic recovery.
        if sha256_file(path)!=before:
            raise ScanError('Source selection changed during upload; local copy retained')
        self.retire_completed_checkpoint(path,result['month'])

    def retire_completed_checkpoint(self,path,month):
        path=Path(path)
        if not path.exists():return
        complete=self.completed[month]
        value=json.loads(path.read_text())
        saved=value.pop('checkpoint_sha256',None)
        if saved!=complete['source_checkpoint_sha256'] or hashlib.sha256(canonical(value).encode()).hexdigest()!=saved:
            raise ScanError('Local candidate cache differs from the verified cloud month; retained')
        path.unlink()

    def _second_build(self,complete):
        reproduce_month(complete,self.config,self.config['frozen_dataset_d_manifest'],self.uploader,
            run_id=self.run_id,audit=self.audit)
        self.completed[complete['month']]=complete
        self.reproduced.add(complete['month'])

    def _catalog(self):
        report=self.audit.summary(self.frozen['position_fens'] if self.d_sha is not None else None)
        report['dataset_d_status']='not_checked_deferred' if self.d_sha is None else 'checked_frozen_snapshot'
        if self.d_sha is not None and any(report[key]['train_dataset_d_shared_unique'] for key in ['all_plies','ply_ge_10']):
            raise ScanError('Frozen Dataset D position leaked into training records')
        expected=dict(train=35000,validation=7500,test=7500)
        all_done=set(self.completed)==set(self.config['months'])
        if all_done and report['games_by_split']!=expected:
            raise ScanError('Final exact split counts disagree with compact audit index')
        # Only complete/reproduced months are advertised by the card/catalog.
        self.catalog=publish_docs(self.uploader,list(self.completed.values()),run_id=self.run_id,
            fully_reproduced=all_done and self.reproduced==set(self.config['months']),audit_report=report)

    def finalize(self):
        if set(self.completed)!=set(self.config['months']):
            raise ScanError('Not all monthly HF artifacts completed')
        self._catalog()
        return self.catalog

    def close(self):self.audit.close()

    def finish_session(self):
        for month in self.config['session_months']:
            if month not in self.completed or month not in self.reproduced:
                raise ScanError('Requested month did not finish upload and reproducibility')
        self._catalog()
        return self.catalog
