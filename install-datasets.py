import requests, zipfile, StringIO
import sys

DEMAND_FORECAST = "demand-forecast"
FRAUD_PREVENTION = "fraud-prevention"
HEARTBEAT_CLASSIFIER = "heartbeat-classifier"

demandForecastURL = "http://download782.mediafire.com/w7ujcex6lwrg/znhjtbd1rc1qz2f/inventory-demand.zip"
fraudPreventionURL = "http://download1643.mediafireuserdownload.com/fe6968gavypg/axsdfq24d7bn26n/creditcardfraud.zip"
heartbeatClassifierURL = "http://download837.mediafire.com/1fcscizx4ygg/dwf9ph4bhx8jhdx/heartbeat-sounds.zip"

tutorialURLs = { DEMAND_FORECAST : demandForecastURL, FRAUD_PREVENTION : fraudPreventionURL, HEARTBEAT_CLASSIFIER : heartbeatClassifierURL}

for tutorialName in tutorialURLs:
    url = tutorialURLs[tutorialName]
    print("Downloading: " + url + " ...")
    response = requests.get(url, stream=True)
    total_length = response.headers.get('content-length')

    if total_length is None: # no content length header
        print("d")
    else:
        dl = 0
        total_length = int(total_length)
        content = StringIO.StringIO()
        for data in response.iter_content(chunk_size=4096):
            dl += len(data)
            content.write(data)
            done = int(50 * dl / total_length)
            sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
            sys.stdout.flush()
    print()
    zipDocument = zipfile.ZipFile(content)
    if tutorialName == DEMAND_FORECAST:
        filenamePath = "./linear-regression/" + DEMAND_FORECAST + "/dataset" 
        zipDocument.extractall(filenamePath)
    elif tutorialName == FRAUD_PREVENTION:
        filenamePath = "./logistic-regression/" + FRAUD_PREVENTION + "/dataset" 
        zipDocument.extractall(filenamePath)
    elif tutorialName == HEARTBEAT_CLASSIFIER:
        filenamePath = "./clustering/" + HEARTBEAT_CLASSIFIER + "/dataset" 
        zipDocument.extractall(filenamePath)