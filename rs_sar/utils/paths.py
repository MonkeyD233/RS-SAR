"""Class to get global paths"""
from pathlib import Path
from rs_sar.utils import RS_SAR_HOME
from rs_sar.utils.configurable import Configurable


class Paths(Configurable):

    def __init__(self):
        self.config_path = RS_SAR_HOME / 'config_files' / 'config.ini'
        super().__init__(self.config_path)
        self.paths = dict()
        self._build()

    def _build(self):
        self.paths['logs'] = Path(self.config['data']['logs'])
        self.paths['RS-SAR'] = Path(self.config['data']['rs_sar'])

    def get(self):
        return self.paths
