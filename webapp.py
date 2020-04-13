# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:06:59 2020

@author: robbf
"""

import functools

from flask import (
        Blueprint, flash, g, redirect, render_template, request, session, url_for
        )

from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug import secure_filename

import pandas as pd

from G1DA.db import get_db
from G1DA.auth import login_required

import os

import PyPDF2 as pdf

from zipfile import ZipFile

from . import ml_predict as mlp

from flask import Flask


bp = Blueprint('main',__name__)

@bp.route('/', methods=('GET','POST'))
@login_required
def index():
    if request.method == 'POST':
        g.table = True
        tasks = []
        titles = []
        
        for file in request.files.getlist('filetoupload'):
            title = secure_filename(file.filename)
            title = title[:-4]
            title = title.split("_")
            title = " ".join(title)
            raw_task_input, raw_title_input = mlp.gather_tasks(file,zip_=False,title=title)
            for task in raw_task_input:
                tasks.append(task)
                titles.append(title)
                
        pd.set_option('display.max_colwidth', -1)
        dataframe = mlp.run_predictions(tasks,titles)
        
        dataframe.sort_values('probability',inplace=True,ascending=False)
        
        dataframe
        
        grouped = dataframe.groupby('IWA_actual_title')
        dataframe_count = grouped.count()
        dataframe_count['Position Time Spent'] = dataframe_count['Position_title'].apply(lambda x: x/sum(dataframe_count['Position_title']))
        dataframe_count.drop(['Position_title','probability'],axis=1,inplace=True)
        
        dataframe_sum = grouped.mean()
        
        dataframe_uniq = grouped.nunique()
        
        dataframe_master = pd.merge(dataframe_count,dataframe_sum,left_index=True,right_index=True)
        dataframe_master = pd.merge(dataframe_master,dataframe_uniq,left_index=True,right_index=True)
        
        dataframe_master.drop(['task_x','IWA_clusterid_x','IWA_actual_id_x','IWA_clusterid_y','probability_x','task_y','IWA_clusterid','IWA_actual_id_y','probability_y','IWA_actual_title'],
                              axis=1,
                              inplace=True)
        
        dataframe_master.columns = ['% of All Tasks','Positions Performing this Task']
        
        g.overview = dataframe_master
        g.task = dataframe
        
        return render_template('task/upload.html', tables=[dataframe_master.to_html(classes='data',header="true")])
    
    g.table = False
    return render_template('task/upload.html')

@bp.route('/register', methods=('GET','POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None
        
        if not username:
            error = 'Username is required'
        elif not password:
            error = 'Password is required'
        elif db.execute (
                'SELECT id FROM user WHERE username = ?', (username,)
                ).fetchone() is not None:
            error = f'User {username} is already registered'
            
        if error is None:
            db.execute(
                    'INSERT INTO user (username, password) VALUES (?,?)',
                    (username,generate_password_hash(password))
                    )
            db.commit()
            return redirect(url_for('auth.login'))
        
        flash(error)
        
    return render_template('auth/register.html')

@bp.route('/login',methods=('GET','POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None
        
        user = db.execute(
                'SELECT * FROM user WHERE username = ?',(username,)
                ).fetchone()
        
        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'],password):
            error = 'Incorrect password.'
        
        if error is None:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('index'))
        
        flash(error)
        
    return render_template('auth/login.html')

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    
    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
                'SELECT * FROM user WHERE id = ?',(user_id,)
                ).fetchone()

@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))
        return view(**kwargs)
    return wrapped_view

@login_required
@bp.route('/upload',methods=('GET','POST'))
def upload():
    if request.method == 'POST':
        file = request.form['filetoupload']
        print(file)
    
    return render_template('task/upload.html')
            
        