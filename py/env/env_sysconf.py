import sysconfig

install_dir = sysconfig.get_paths()['stdlib']
print(install_dir)
