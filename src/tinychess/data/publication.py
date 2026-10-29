"""Post-publication consistency checks, independent of the upload writer."""


def check_object(info, expected):
    """Verify final-revision LFS content IDs; regular blobs need a full read."""
    if info.path != expected['path_in_repo'] or info.size != expected['bytes']:
        raise ValueError('Final published object path/size mismatch')
    if info.lfs is None:
        return False
    if info.lfs.size != expected['bytes'] or info.lfs.sha256 != expected['sha256']:
        raise ValueError('Final published LFS SHA256 mismatch')
    return True


def check_card_paths(metadata, expected):
    configs = {config['config_name']: config for config in metadata['configs']}
    if set(configs) != {'human_v3', 'games'} or not configs['human_v3']['default']:
        raise ValueError('Unexpected dataset configurations or default')
    for kind, config_name in [('positions', 'human_v3'), ('games', 'games')]:
        config = configs[config_name]
        actual = {entry['split']: entry['path'] for entry in config['data_files']}
        if actual != expected[kind]:
            raise ValueError('Published card paths disagree with verified monthly manifests')
