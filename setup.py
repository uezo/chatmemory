from setuptools import setup

setup(
    name="chatmemory",
    version="0.2.2",
    url="https://github.com/uezo/chatmemory",
    author="uezo",
    author_email="uezo@uezo.net",
    maintainer="uezo",
    maintainer_email="uezo@uezo.net",
    description="The simple yet powerful long-term memory manager between AI and you💕",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=["fastapi==0.115.8", "openai==1.64.0", "uvicorn==0.34.0", "psycopg2-binary==2.9.10"],
    license="Apache v2",
    packages=["chatmemory"],
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
