from modules.runExp import runExp
from configs.getConfig import getConfig

resultPath = '../result'
configName = 'debug'
config = getConfig(configName)
runExp(config, configName, resultPath)