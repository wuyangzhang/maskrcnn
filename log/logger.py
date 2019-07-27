import logging

logging.basicConfig(filename='./trace.log',
                    format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.WARN,
                    )

#logging.Formatter(fmt='%(asctime)s.%(msecs)03d',datefmt='%Y-%m-%d,%H:%M:%S')

mylogger = logging.getLogger('mobidist')
