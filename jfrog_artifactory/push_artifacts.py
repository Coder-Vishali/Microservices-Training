import os
from artifactory import ArtifactoryPath
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

path = ArtifactoryPath("https://<user>:<pass>@<artifactory_server_name>/<repository_name>/", verify=False)
path.deploy_file('./sample.txt')

folder_path=<folder location>
for root, dirs, files in os.walk(folder_path, topdown=False):
  for name in files:
      print(f"Deploying {os.path.join(folder_path,name)} ")
      path.mkdir(model_path)
      path.deploy_file(os.path.join(folder_path,name))
