import os
from artifactory import ArtifactoryPath
import subprocess

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# Method 1:
path = ArtifactoryPath(
    "https://<user>:<pass>@<artifactory_server_name>/<repository_name>", verify=False)
with path.open() as fd, open("sample.txt", "wb") as out:
    out.write(fd.read())
    
# Method 2:
command = "curl --insecure -u '<user>:<pass>'  -O 'https://<artifactory_server_name>/<repository_name>/sample.txt'"
proc = subprocess.call(command, shell=True)
