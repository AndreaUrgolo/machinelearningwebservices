#Author: Andrea Urgolo
from flask import session, request, url_for


""" Utility functions """        

def redirect_url(default='index'):
    return request.args.get('next') or \
           request.referrer or \
           url_for(default)

def clear_session(clear_all=False):
    if clear_all:
        session.clear()
    else :# leaves flash messages
        [session.pop(key) for key in list(session.keys()) if key != '_flashes']
