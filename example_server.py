import logging
from chatmemory.server import ChatMemoryServer

OPENAI_APIKEY = "YOUR_API_KEY"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter("%(asctime)s %(levelname)8s %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(log_format)
logger.addHandler(streamHandler)

logger.info("starting sever...")

server = ChatMemoryServer(openai_apikey=OPENAI_APIKEY)
server.start()
