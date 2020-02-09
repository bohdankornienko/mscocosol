import os
import sys


if 'linux' == sys.platform:
    print('Registring repository for Linux system...')

    this_file = os.path.abspath(__file__)
    print('This file absolute path:', this_file)
    this_dir = os.path.split(this_file)[0]
    print('This file directory:', this_dir)

    home = os.environ['HOME']
    bashrc = os.path.join(home, '.bashrc')

    with open(bashrc, 'r') as fp:
        content = fp.readlines()

    line = 'export PYTHONPATH={}:$PYTHONPATH\n'.format(this_dir)

    content += [line, '\n']

    print('Appending .bashrc...')
    with open(bashrc, 'w') as fp:
        fp.writelines(content)

    print('Registration commplete.')
