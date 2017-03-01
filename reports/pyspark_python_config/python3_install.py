from fabric import *


@task
def apt_install():
    sudo('apt-get install python3')


@task
def yum_install():
    sudo('yum install python3')
