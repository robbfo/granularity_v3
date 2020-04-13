# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:10:37 2020

@author: robbf
"""

from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()