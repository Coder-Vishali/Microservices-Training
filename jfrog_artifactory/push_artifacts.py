import os
from artifactory import ArtifactoryPath
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

path = ArtifactoryPath("https://<user>:<pass>@<artifactory_server_name>/<repository_name>/", verify=False)
path.deploy_file('./sample.txt')
