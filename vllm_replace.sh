site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./verl/utils/vllm_replace/* $site_pkg_path/vllm/
