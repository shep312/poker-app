import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or '2435275646847356'