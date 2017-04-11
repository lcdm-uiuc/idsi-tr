from fabric.api import *
import uuid


@task
def install_requirements(requirements_path='requirements.txt'):
    folder = uuid.uuid1()
    with cd('/tmp'):
        # Create a temporary folder
        run('mkdir {}'.format(folder))
        # Upload our requirements.txt
        put(requirements_path, str(folder))
        # Run pip install
        sudo('pip install -r {0}/requirements.txt -U'.format(folder))
        # Delete the temporary files
        run('rm -r {0}'.format(folder))


@task
def update_requirements(requirements_path='requirements.txt'):
    local('pip install virtualenv')
    venv = uuid.uuid1()
    local('virtualenv {0}'.format(venv))
    with prefix('source {0}/bin/activate'.format(venv)):
        local('pip install -r {0} -U'.format(requirements_path))
        local('pip freeze > {0}'.format(requirements_path))
    local('rm -r {0}'.format(venv))
