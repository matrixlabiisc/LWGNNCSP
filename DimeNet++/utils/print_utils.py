# -*- coding: utf-8 -*-
# ========================================================================
#  2021/04/28 下午 04:08
#                 _____   _   _   _____   __   _   _____  
#                /  ___| | | | | | ____| |  \ | | /  ___| 
#                | |     | |_| | | |__   |   \| | | |     
#                | |     |  _  | |  __|  | |\   | | |  _  
#                | |___  | | | | | |___  | | \  | | |_| | 
#                \_____| |_| |_| |_____| |_|  \_| \_____/ 
# ------------------------------------------------------------------------
# 
# 
# 
# ========================================================================

import time


def header():
    print(
        '''
          ______   __    __           ______    ______  
         /      \ /  \  /  |         /      \  /      \ 
        /$$$$$$  |$$  \ $$ |        /$$$$$$  |/$$$$$$  |
        $$ | _$$/ $$$  \$$ | ______ $$ |  $$ |$$ |__$$ |
        $$ |/    |$$$$  $$ |/      |$$ |  $$ |$$    $$ |
        $$ |$$$$ |$$ $$ $$ |$$$$$$/ $$ |  $$ |$$$$$$$$ |
        $$ \__$$ |$$ |$$$$ |        $$ \__$$ |$$ |  $$ |
        $$    $$/ $$ | $$$ |        $$    $$/ $$ |  $$ |
         $$$$$$/  $$/   $$/          $$$$$$/  $$/   $$/ 
        '''
    )
    print('\tProgram: GN-BOSS\tVersion: 0.2.0\t')
    print(time.strftime("%b %d %Y %H:%M:%S", time.localtime()))
    print('')


def print_header(func):
    def wrapper(*args, **kwargs):
        header()
        return func(*args, **kwargs)

    return wrapper


def print_run_info(info):
    def wrapper(func):
        def _wrapper(*args, **kargs):
            print(info + '! Running ...')
            res = func(*args, **kargs)
            print(info + ' OK!')
            print('')
            return res

        return _wrapper

    return wrapper


if __name__ == '__main__':
    header()
