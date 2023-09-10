import os
from argparse import ArgumentParser
import logging
from .server import ChatMemoryServer

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s %(levelname)8s %(message)s")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    usage = "Usage: python {} [--key <openai api key>] [--host <ip address or hostname>] [--port <port_number>] [--help]".format(__file__)
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument("--key", dest="api_key", help="OpenAI API Key. Set here or environment variables with key 'OPENAI_APIKEY'")
    argparser.add_argument("--host", dest="host", default="127.0.0.1", help="IP address to listen. Default is 127.0.0.1")
    argparser.add_argument("--port", dest="port", default=8123, type=int, help="Port numbert to listen. Default is 8123")
    args = argparser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_APIKEY")
    if not api_key:
        logger.error("OpenAI API Key is missing")
        return

    host = args.host
    port = args.port

    logger.info("starting sever...")

    server = ChatMemoryServer(openai_apikey=api_key)
    server.start(host=host, port=port)


if __name__ == "__main__":
    main()
