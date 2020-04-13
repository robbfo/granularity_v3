# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:12:41 2020

@author: robbf
"""


import os

from flask import Flask

def create_app(test_config=None):
    #create and configure the app
    app = Flask(__name__,
                instance_relative_config=True)
    app.config.from_mapping(
            SECRET_KEY='dev',
            DATABASE=os.path.join(app.instance_path,'flaskr.sprite'),
            )
    
    if test_config is None:
        #load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py',silent=True)
    else:
        #load the test config if passed in
        app.config.from_mapping(test_config)
        
    #ensure instance folder exists
    try: 
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    from . import db
    db.init_app(app)
    
    from . import auth
    app.register_blueprint(auth.bp)
    
    from . import taskview
    app.register_blueprint(taskview.bp)
    app.add_url_rule('/',endpoint='index')
    
    from . import taskdash
    app.register_blueprint(taskdash.bp)
    app.add_url_rule('/dash',endpoint='dashboard')

    return app
