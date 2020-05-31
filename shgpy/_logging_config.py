import logging
import pathlib
LOGFILE = pathlib.Path(__name__).parent / 'log' / 'shgpy.log'

cfg = {
    'version':1,
    'formatters':{
        'f':{
            'format':'%(asctime)s shgpy %(levelname)-8s %(message)s',
        },
    },
    'handlers':{
        'stream':{
            'class':'logging.StreamHandler',
            'formatter':'f',
            'level':logging.INFO,
        },
        'file':{
            'class':'logging.FileHandler',
            'formatter':'f',
            'level':logging.DEBUG,
            'filename':LOGFILE,
            'mode':'w',
        }
    },
    'root':{
        'handlers':['stream', 'file'],
        'level':logging.DEBUG,
    },
}
