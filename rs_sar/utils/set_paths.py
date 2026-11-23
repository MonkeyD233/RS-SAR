"""Script to set the RS-SAR dataset and logs paths in config.ini"""
import argparse
from rs_sar.utils.configurable import Configurable
from rs_sar.utils import RS_SAR_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--rs-sar', default='/rs_sar',
                        help='Path to the RS-SAR dataset root.')
    parser.add_argument('--logs', default='/root/workspace/logs',
                        help='Path where logs and models are saved.')
    args = parser.parse_args()
    config_path = RS_SAR_HOME / 'config_files' / 'config.ini'
    configurable = Configurable(config_path)
    configurable.set('data', 'rs_sar', args.rs_sar)
    # configurable.set('data', 'pdr', args.pdr)
    configurable.set('data', 'logs', args.logs)
    with open(config_path, 'w') as fp:
        configurable.config.write(fp)
